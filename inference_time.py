import math
import shutil
import statistics
import sys
from pathlib import Path
from typing import Any, Literal

import delu
import numpy as np
import rtdl_num_embeddings
import scipy
import torch
import torch.nn as nn
import torch.utils.tensorboard
from loguru import logger
from torch import Tensor
from tqdm import tqdm
from typing_extensions import NotRequired, TypedDict
import time
import pickle

if __name__ == '__main__':
    _cwd = Path.cwd()
    assert _cwd.joinpath(
        '.git'
    ).exists(), 'The script must be run from the root of the repository'
    sys.path.append(str(_cwd))
    del _cwd

import lib
import lib.data
import lib.deep
import lib.env
from lib import KWArgs, PartKey


class Model(nn.Module):
    """MLP & TabM."""

    def __init__(
            self,
            *,
            n_num_features: int,
            cat_cardinalities: list[int],
            n_classes: None | int,
            backbone: dict,
            bins: None | list[Tensor],  # For piecewise-linear encoding/embeddings.
            num_embeddings: None | dict = None,
            arch_type: Literal[
                # Plain feed-forward network without any kind of ensembling.
                'plain',
                    #
                    # TabM-mini
                'tabm-mini',
                    #
                    # TabM-mini. The first adapter is initialized from the normal distribution.
                    # This is used in Section 5.1.
                'tabm-mini-normal',
                    #
                    # TabM
                'tabm',
                    #
                    # TabM. The first adapter is initialized from the normal distribution.
                    # This variation was is not used in the paper, but there is a preliminary
                    # evidence that may be a better default strategy.
                'tabm-normal',
            ],
            k: None | int = None,
    ) -> None:
        # >>> Validate arguments.
        assert n_num_features >= 0
        assert n_num_features or cat_cardinalities
        if arch_type == 'plain':
            assert k is None
        else:
            assert k is not None
            assert k > 0

        super().__init__()

        # >>> Continuous (numerical) features
        first_adapter_sections = []  # See the comment in `_init_first_adapter`.

        if n_num_features == 0:
            assert bins is None
            self.num_module = None
            d_num = 0

        elif num_embeddings is None:
            assert bins is None
            self.num_module = None
            d_num = n_num_features
            first_adapter_sections.extend(1 for _ in range(n_num_features))

        else:
            if bins is None:
                self.num_module = lib.deep.make_module(
                    **num_embeddings, n_features=n_num_features
                )
            else:
                assert num_embeddings['type'].startswith('PiecewiseLinearEmbeddings')
                self.num_module = lib.deep.make_module(**num_embeddings, bins=bins)
            d_num = n_num_features * num_embeddings['d_embedding']
            first_adapter_sections.extend(
                num_embeddings['d_embedding'] for _ in range(n_num_features)
            )

        # >>> Categorical features
        self.cat_module = (
            lib.deep.OneHotEncoding0d(cat_cardinalities) if cat_cardinalities else None
        )
        first_adapter_sections.extend(cat_cardinalities)
        d_cat = sum(cat_cardinalities)

        # >>> Backbone
        d_flat = d_num + d_cat
        self.minimal_ensemble_adapter = None
        if backbone['type'] in ['BMoE', 'BMoIE']:
            d_out = 1 if n_classes is None else n_classes
        else:
            d_out = None
        self.backbone = lib.deep.make_module(d_in=d_flat, **backbone, d_out=d_out)

        if arch_type != 'plain':
            assert k is not None
            first_adapter_init = (
                'normal'
                if arch_type in ('tabm-mini-normal', 'tabm-normal')
                # For other arch_types, the initialization depends
                # on the presense of num_embeddings.
                else 'random-signs'
                if num_embeddings is None
                else 'normal'
            )

            if arch_type in ('tabm-mini', 'tabm-mini-normal'):
                # Minimal ensemble
                self.minimal_ensemble_adapter = lib.deep.ScaleEnsemble(
                    k,
                    d_flat,
                    init='random-signs' if num_embeddings is None else 'normal',
                )
                _init_first_adapter(
                    self.minimal_ensemble_adapter.weight,  # type: ignore[code]
                    first_adapter_init,
                    first_adapter_sections,
                )

            elif arch_type in ('tabm', 'tabm-normal'):
                # Like BatchEnsemble, but all multiplicative adapters,
                # except for the very first one, are initialized with ones.
                lib.deep.make_efficient_ensemble(
                    self.backbone,
                    k=k,
                    ensemble_scaling_in=True,
                    ensemble_scaling_out=True,
                    ensemble_bias=True,
                    scaling_init='ones',
                )
                _init_first_adapter(
                    _get_first_ensemble_layer(self.backbone).r,  # type: ignore[code]
                    first_adapter_init,
                    first_adapter_sections,
                )

            else:
                raise ValueError(f'Unknown arch_type: {arch_type}')

        # >>> Output

        if backbone['type'] in ['BMoE', 'BMoIE']:
            self.output = None
        else:
            d_block = backbone['d_block']
            d_out = 1 if n_classes is None else n_classes
            self.output = (
                nn.Linear(d_block, d_out)
                if arch_type == 'plain'
                else lib.deep.NLinear(k, d_block, d_out)  # type: ignore[code]
            )

        # >>>
        self.arch_type = arch_type
        self.k = k

    def forward(
            self, x_num: None | Tensor = None, x_cat: None | Tensor = None,
            num_samples: None | int = None, return_average: None | bool = None,
    ) -> Tensor:
        x = []
        if x_num is not None:
            x.append(x_num if self.num_module is None else self.num_module(x_num))
        if x_cat is None:
            assert self.cat_module is None
        else:
            assert self.cat_module is not None
            x.append(self.cat_module(x_cat).float())
        x = torch.column_stack([x_.flatten(1, -1) for x_ in x])

        if self.k is not None:
            x = x[:, None].expand(-1, self.k, -1)  # (B, D) -> (B, K, D)
            if self.minimal_ensemble_adapter is not None:
                x = self.minimal_ensemble_adapter(x)
        else:
            assert self.minimal_ensemble_adapter is None

        if (return_average is not None) and (num_samples is not None):
            x = self.backbone(x, num_samples=num_samples, return_average=return_average)
        else:
            x = self.backbone(x)
        if self.output is not None:
            x = self.output(x)
        if self.k is None:
            # Adjust the output shape for plain networks to make them compatible
            # with the rest of the script (loss, metrics, predictions, ...).
            # (B, D_OUT) -> (B, 1, D_OUT)
            # for bayesian ensembles (N, B, D_OUT) -> (N, B, 1, D_OUT)
            if return_average is False:  # can not be simplified, because it might be None
                x = x[:, :, None]
            else:
                x = x[:, None]
        return x


class Config(TypedDict):
    seed: int
    data: KWArgs
    bins: NotRequired[KWArgs]
    model: KWArgs
    head_selection: NotRequired[bool]
    optimizer: KWArgs
    n_lr_warmup_epochs: NotRequired[int]
    batch_size: int
    eval_batch_size: NotRequired[int]
    patience: int
    n_epochs: int
    gradient_clipping_norm: NotRequired[float]
    parameter_statistics: NotRequired[bool]
    n_bayesian_ensembles: NotRequired[int]
    num_samples: NotRequired[int]
    return_average: NotRequired[bool]
    # NOTE
    # Please, read these notes before using AMP and/or `torch.compile`.
    #
    # The usage of the following efficiency-related settings depends on the model.
    # To learn if a given model can run with AMP and torch.compile on a given task,
    # try activating these settings and check if the task metrics are satisfactory.
    # The following notes can be helpful.
    #
    # - For simple architectures, such as MLP or TabM, these settings often
    #   make models significantly faster without any negative side-effects.
    #   For a real world task, it is worth to doublecheck that by comparing runs
    #   with and without AMP and/or torch.compile.
    #
    # - For more complex architectures, these settings should be used
    #   with extra caution. For example, some baselines used in this project showed
    #   worse performance when trained with AMP. For some models, AMP with BF16 hurts
    #   the performance, but AMP with FP16 works fine. Sometimes, it is the opposite.
    #   Sometimes, it depends on a dataset. Because of that, all baselines were run
    #   without AMP and torch.compile to ensure that results are representative.
    #
    # - AMP usually provides significantly larger speedups than `torch.compile`.
    #   So, if there are any issues with `torch.compile`, using only AMP will still
    #   lead to substantially faster models.
    #
    # - If a training run is already fast (e.g. on small datasets),
    #   `torch.compile` can make it *slower*, because the compilation itself
    #   takes some time (in particular, at the beginning of the first epoch,
    #   and at the beginning of the first evaluation).
    #
    # - Generally, compared to AMP, `torch.compile` is a younger technology, and a
    #   model must meet certain requirements to be compatible with `torch.compile`.
    #   In case of any issues, try updating PyTorch.
    amp: NotRequired[bool]  # torch.autocast
    compile: NotRequired[bool]  # torch.compile


def main(
        config: Config | str | Path,
        output: None | str | Path = None,
        *,
        force: bool = False,
        num_samples: None | int
        # ) -> None | lib.JSONDict:
) -> float:
    # >>> Start
    config, output = lib.check(config, output, config_type=Config)
    # if not lib.start(output, force=force):
    #     return None

    # lib.print_config(config)  # type: ignore[code]
    delu.random.seed(config['seed'])
    device = lib.get_device()
    report = lib.create_report(main, config)

    # >>> Data
    dataset = lib.data.build_dataset(**config['data'])
    if dataset.task.is_regression:
        dataset.data['y'], regression_label_stats = lib.data.standardize_labels(
            dataset.data['y']
        )
    else:
        regression_label_stats = None

    # Convert binary features to categorical features.
    if dataset.n_bin_features > 0:
        x_bin = dataset.data.pop('x_bin')
        # Remove binary features with just one unique value in the training set.
        # This must be done, otherwise, the script will fail on one specific dataset
        # from the "why" benchmark.
        n_bin_features = x_bin['train'].shape[1]
        good_bin_idx = [
            i for i in range(n_bin_features) if len(np.unique(x_bin['train'][:, i])) > 1
        ]
        if len(good_bin_idx) < n_bin_features:
            x_bin = {k: v[:, good_bin_idx] for k, v in x_bin.items()}

        if dataset.n_cat_features == 0:
            dataset.data['x_cat'] = {
                part: np.zeros((dataset.size(part), 0), dtype=np.int64)
                for part in x_bin
            }
        for part in x_bin:
            dataset.data['x_cat'][part] = np.column_stack(
                [dataset.data['x_cat'][part], x_bin[part].astype(np.int64)]
            )
        del x_bin
    dataset = dataset.to_torch(device)
    Y_train = dataset.data['y']['train'].to(
        torch.long if dataset.task.is_classification else torch.float
    )

    # >>> Model
    if 'bins' in config:
        # Compute the bins for PiecewiseLinearEncoding and PiecewiseLinearEmbeddings.
        compute_bins_kwargs = (
            {
                'y': Y_train.to(
                    torch.long if dataset.task.is_classification else torch.float
                ),
                'regression': dataset.task.is_regression,
                'verbose': True,
            }
            if 'tree_kwargs' in config['bins']
            else {}
        )
        bin_edges = rtdl_num_embeddings.compute_bins(
            dataset.data['x_num']['train'], **config['bins'], **compute_bins_kwargs
        )
        logger.info(f'Bin counts: {[len(x) - 1 for x in bin_edges]}')
    else:
        bin_edges = None
    model = Model(
        n_num_features=dataset.n_num_features,
        cat_cardinalities=dataset.compute_cat_cardinalities(),
        n_classes=dataset.task.try_compute_n_classes(),
        **config['model'],
        bins=bin_edges,
    )
    report['n_parameters'] = lib.deep.get_n_parameters(model)
    logger.info(f'n_parameters = {report["n_parameters"]}')
    report['prediction_type'] = 'labels' if dataset.task.is_regression else 'probs'
    model.to(device)
    if lib.is_dataparallel_available():
        model = nn.DataParallel(model)

    # >>> Training
    step = 0
    batch_size = config['batch_size']
    report['epoch_size'] = epoch_size = math.ceil(dataset.size('train') / batch_size)
    eval_batch_size = config.get(
        'eval_batch_size',
        # With torch.compile,
        # the largest possible evaluation batch size is noticeably smaller.
        2048 if config.get('compile', False) else 32768,
    )
    chunk_size = None
    optimizer = lib.deep.make_optimizer(
        **config['optimizer'], params=lib.deep.make_parameter_groups(model)
    )
    gradient_clipping_norm = config.get('gradient_clipping_norm')
    _loss_fn = (
        nn.functional.mse_loss
        if dataset.task.is_regression
        else nn.functional.cross_entropy
    )

    def loss_fn(y_pred: Tensor, y_true: Tensor) -> Tensor:
        return _loss_fn(y_pred.flatten(0, 1), y_true.repeat_interleave(y_pred.shape[1]))

    # The following generator is used only for creating training batches,
    # so the random seed fully determines the sequence of training objects.
    batch_generator = torch.Generator(device).manual_seed(config['seed'])
    timer = delu.tools.Timer()
    early_stopping = delu.tools.EarlyStopping(config['patience'], mode='max')
    parameter_statistics = config.get('parameter_statistics', config['seed'] == 1)
    training_log = []
    # writer = torch.utils.tensorboard.SummaryWriter(output)  # type: ignore[code]

    # Only bfloat16 was tested as amp_dtype.
    # However, float16 is supported as a fallback.
    # To enable float16, uncomment the two lines below.
    amp_dtype = (
        torch.bfloat16
        if config.get('amp', False)
           and torch.cuda.is_available()
           and torch.cuda.is_bf16_supported()
        # else torch.float16
        # if config.get('amp', False) and and torch.cuda.is_available()
        else None
    )
    amp_enabled = amp_dtype is not None
    # For FP16, the gradient scaler must be used.
    grad_scaler = torch.cuda.amp.GradScaler() if amp_dtype is torch.float16 else None  # type: ignore[code]
    logger.info(f'AMP enabled: {amp_enabled}')

    if config.get('compile', False):
        # NOTE
        # `torch.compile` is intentionally called without the `mode` argument,
        # because it caused issues with training.
        model = torch.compile(model)
        evaluation_mode = torch.no_grad
    else:
        evaluation_mode = torch.inference_mode

    @torch.autocast(device.type, enabled=amp_enabled, dtype=amp_dtype)  # type: ignore[code]
    def apply_model(part: PartKey, idx: Tensor, return_average=None, num_samples=None) -> Tensor:
        return (
            model(
                dataset.data['x_num'][part][idx] if 'x_num' in dataset.data else None,
                dataset.data['x_cat'][part][idx] if 'x_cat' in dataset.data else None,
                num_samples=num_samples, return_average=return_average
            )
            .squeeze(-1)  # Remove the last dimension for regression predictions.
            .float()
        )

    @evaluation_mode()
    def evaluate(
            parts: list[PartKey], eval_batch_size: int, return_average: bool = True, num_samples: None | int = None
    ) -> float:
        # ) -> tuple[
        #     dict[PartKey, Any], dict[PartKey, np.ndarray], dict[PartKey, np.ndarray], int
        # ]:
        model.eval()
        head_predictions: dict[PartKey, np.ndarray] = {}
        for part in parts:
            while eval_batch_size:
                try:

                    cat_dim = 0 if return_average else 1
                    start_time = time.time()
                    head_predictions[part] = (
                        torch.cat(
                            [
                                apply_model(part, idx, return_average=return_average,
                                            num_samples=num_samples)
                                for idx in torch.arange(
                                dataset.size(part), device=device
                            ).split(eval_batch_size)
                            ], dim=cat_dim
                        )
                        .cpu()
                        .numpy()
                    )
                    end_time = time.time()

                except RuntimeError as err:
                    if not lib.is_oom_exception(err):
                        raise
                    eval_batch_size //= 2
                    logger.warning(f'eval_batch_size = {eval_batch_size}')
                else:
                    break
            if not eval_batch_size:
                RuntimeError('Not enough memory even for eval_batch_size=1')
        return end_time - start_time
        # if not return_average:
        #     metrics = {}
        #     predictions = {}
        #     n_bayesian_ensembles = config.get('n_bayesian_ensembles', None)
        #     for n in n_bayesian_ensembles:
        #         if dataset.task.is_regression:
        #             assert regression_label_stats is not None
        #             n_predictions = {
        #                 k: v[:n, ...].mean(axis=0) * regression_label_stats.std + regression_label_stats.mean
        #                 for k, v in head_predictions.items()
        #             }
        #         else:
        #             n_predictions = {
        #                 k: scipy.special.softmax(v[:n, ...].mean(axis=0), axis=-1)
        #                 for k, v in head_predictions.items()
        #             }
        #
        #             if dataset.task.is_binclass:
        #                 n_predictions = {k: v[..., 1] for k, v in n_predictions.items()}
        #
        #         n_predictions = {k: v.mean(1) for k, v in n_predictions.items()}
        #         predictions[str(n)] = n_predictions
        #         metrics[str(n)] = (
        #             dataset.task.calculate_metrics(n_predictions, report['prediction_type'])
        #             if lib.are_valid_predictions(predictions)
        #             else {x: {'score': lib.WORST_SCORE} for x in predictions}
        #         )
        # else:
        #     if dataset.task.is_regression:
        #         assert regression_label_stats is not None
        #         head_predictions = {
        #             k: v * regression_label_stats.std + regression_label_stats.mean
        #             for k, v in head_predictions.items()
        #         }
        #     else:
        #         head_predictions = {
        #             k: scipy.special.softmax(v, axis=-1)
        #             for k, v in head_predictions.items()
        #         }
        #         if dataset.task.is_binclass:
        #             head_predictions = {k: v[..., 1] for k, v in head_predictions.items()}
        #
        #     predictions = {k: v.mean(1) for k, v in head_predictions.items()}
        #     metrics = (
        #         dataset.task.calculate_metrics(predictions, report['prediction_type'])
        #         if lib.are_valid_predictions(predictions)
        #         else {x: {'score': lib.WORST_SCORE} for x in predictions}
        #     )
        # return metrics, predictions, head_predictions, eval_batch_size

    return evaluate(
        ['train', 'val', 'test'], eval_batch_size, num_samples=num_samples
    )


import glob
import os


def find_files_with_pattern(base_path, pattern="0-evaluation/0.toml"):
    search_pattern = os.path.join(base_path, "**", pattern)
    return glob.glob(search_pattern, recursive=True)


# Example usage:
base_directories = [
    "exp/results/gumbel_evaluation_results/bmoe",
    "exp/results/gumbel_evaluation_results/bmoe-piecewiselinear",

    "exp/results/evaluation_results_16_04_2024/moe",
    "exp/results/evaluation_results_16_04_2024/moe-piecewiselinear",
    "exp/mlp/",
    "exp/mlp-piecewiselinear/",

]  # Change this to your starting directory
matching_files = []
for dir in base_directories:
    matching_files += find_files_with_pattern(dir)

final_files = []
for f in matching_files:
    if 'tabred' not in f:
        final_files.append(f)
    else:
        print('excluded:', f)

# Print found files

import glob
import os


def find_files_with_pattern(base_path, pattern="0-evaluation/0.toml"):
    search_pattern = os.path.join(base_path, "**", pattern)
    return glob.glob(search_pattern, recursive=True)


num_samples = [5, 10, 100]
n = 15
# import numpy as np
res = {'1': {}, '5': {}, '10': {}, '100': {}}
# Print found files
for file in final_files:
    exec_time = []
    for i in range(n):
        exec_time.append(main(file, num_samples=None))
    res['1'][file] = exec_time

for n in num_samples:
    print(f'current number of samples:{n}')
    for file in final_files:
        if 'bmoe' in file:
            exec_time = []
            for i in range(n):
                exec_time.append(main(file, num_samples=n))
            res[str(n)][file] = exec_time
    # print(file.split('/'))
# print(res)
# Save to a pickle file
with open("execution_time.pkl", "wb") as file:
    pickle.dump(res, file)
