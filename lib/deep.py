import itertools
from collections.abc import Callable
from typing import Any, Literal, cast

import delu
import rtdl_num_embeddings
import rtdl_revisiting_models
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from .util import is_oom_exception
from .bnn import BayesianGatingNetwork


# ======================================================================================
# Initialization
# ======================================================================================
def init_rsqrt_uniform_(x: Tensor, d: int) -> Tensor:
    assert d > 0
    d_rsqrt = d ** -0.5
    return nn.init.uniform_(x, -d_rsqrt, d_rsqrt)


@torch.inference_mode()
def init_random_signs_(x: Tensor) -> Tensor:
    return x.bernoulli_(0.5).mul_(2).add_(-1)


# ======================================================================================
# Modules
# ======================================================================================
class Identity(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x


class NLinear(nn.Module):
    """A stack of N linear layers. Each layer is applied to its own part of the input.

    **Shape**

    - Input: ``(B, N, in_features)``
    - Output: ``(B, N, out_features)``

    The i-th linear layer is applied to the i-th matrix of the shape (B, in_features).

    Technically, this is a simplified version of delu.nn.NLinear:
    https://yura52.github.io/delu/stable/api/generated/delu.nn.NLinear.html.
    The difference is that this layer supports only 3D inputs
    with exactly one batch dimension. By contrast, delu.nn.NLinear supports
    any number of batch dimensions.
    """

    def __init__(
            self, n: int, in_features: int, out_features: int, bias: bool = True
    ) -> None:
        super().__init__()
        self.weight = Parameter(torch.empty(n, in_features, out_features))
        self.bias = Parameter(torch.empty(n, out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        d = self.weight.shape[-2]
        init_rsqrt_uniform_(self.weight, d)
        if self.bias is not None:
            init_rsqrt_uniform_(self.bias, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 3
        assert x.shape[-(self.weight.ndim - 1):] == self.weight.shape[:-1]

        x = x.transpose(0, 1)
        x = x @ self.weight
        x = x.transpose(0, 1)
        if self.bias is not None:
            x = x + self.bias
        return x


class PiecewiseLinearEmbeddings(rtdl_num_embeddings.PiecewiseLinearEmbeddings):
    """
    This class simply adds the default values for `activation` and `version`.
    """

    def __init__(
            self,
            *args,
            activation: bool = False,
            version: None | Literal['A', 'B'] = 'B',
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs, activation=activation, version=version)


class OneHotEncoding0d(nn.Module):
    # Input:  (*, n_cat_features=len(cardinalities))
    # Output: (*, sum(cardinalities))

    def __init__(self, cardinalities: list[int]) -> None:
        super().__init__()
        self._cardinalities = cardinalities

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim >= 1
        assert x.shape[-1] == len(self._cardinalities)

        return torch.cat(
            [
                # NOTE
                # This is a quick hack to support out-of-vocabulary categories.
                #
                # Recall that lib.data.transform_cat encodes categorical features
                # as follows:
                # - In-vocabulary values receive indices from `range(cardinality)`.
                # - All out-of-vocabulary values (i.e. new categories in validation
                #   and test data that are not presented in the training data)
                #   receive the index `cardinality`.
                #
                # As such, the line below will produce the standard one-hot encoding for
                # known categories, and the all-zeros encoding for unknown categories.
                # This may not be the best approach to deal with unknown values,
                # but should be enough for our purposes.
                F.one_hot(x[..., i], cardinality + 1)[..., :-1]
                for i, cardinality in enumerate(self._cardinalities)
            ],
            -1,
        )


class ScaleEnsemble(nn.Module):
    def __init__(
            self,
            k: int,
            d: int,
            *,
            init: Literal['ones', 'normal', 'random-signs'],
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(k, d))
        self._weight_init = init
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self._weight_init == 'ones':
            nn.init.ones_(self.weight)
        elif self._weight_init == 'normal':
            nn.init.normal_(self.weight)
        elif self._weight_init == 'random-signs':
            init_random_signs_(self.weight)
        else:
            raise ValueError(f'Unknown weight_init: {self._weight_init}')

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim >= 2
        return x * self.weight


class LinearEfficientEnsemble(nn.Module):
    """
    This layer is a more configurable version of the "BatchEnsemble" layer
    from the paper
    "BatchEnsemble: An Alternative Approach to Efficient Ensemble and Lifelong Learning"
    (link: https://arxiv.org/abs/2002.06715).

    First, this layer allows to select only some of the "ensembled" parts:
    - the input scaling  (r_i in the BatchEnsemble paper)
    - the output scaling (s_i in the BatchEnsemble paper)
    - the output bias    (not mentioned in the BatchEnsemble paper,
                          but is presented in public implementations)

    Second, the initialization of the scaling weights is configurable
    through the `scaling_init` argument.

    NOTE
    The term "adapter" is used in the TabM paper only to tell the story.
    The original BatchEnsemble paper does NOT use this term. So this class also
    avoids the term "adapter".
    """

    r: None | Tensor
    s: None | Tensor
    bias: None | Tensor

    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            *,
            k: int,
            ensemble_scaling_in: bool,
            ensemble_scaling_out: bool,
            ensemble_bias: bool,
            scaling_init: Literal['ones', 'random-signs'],
    ):
        assert k > 0
        if ensemble_bias:
            assert bias
        super().__init__()

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.register_parameter(
            'r',
            (
                nn.Parameter(torch.empty(k, in_features))
                if ensemble_scaling_in
                else None
            ),  # type: ignore[code]
        )
        self.register_parameter(
            's',
            (
                nn.Parameter(torch.empty(k, out_features))
                if ensemble_scaling_out
                else None
            ),  # type: ignore[code]
        )
        self.register_parameter(
            'bias',
            (
                nn.Parameter(torch.empty(out_features))  # type: ignore[code]
                if bias and not ensemble_bias
                else nn.Parameter(torch.empty(k, out_features))
                if ensemble_bias
                else None
            ),
        )

        self.in_features = in_features
        self.out_features = out_features
        self.k = k
        self.scaling_init = scaling_init

        self.reset_parameters()

    def reset_parameters(self):
        init_rsqrt_uniform_(self.weight, self.in_features)
        scaling_init_fn = {'ones': nn.init.ones_, 'random-signs': init_random_signs_}[
            self.scaling_init
        ]
        if self.r is not None:
            scaling_init_fn(self.r)
        if self.s is not None:
            scaling_init_fn(self.s)
        if self.bias is not None:
            bias_init = torch.empty(
                # NOTE: the shape of bias_init is (out_features,) not (k, out_features).
                # It means that all biases have the same initialization.
                # This is similar to having one shared bias plus
                # k zero-initialized non-shared biases.
                self.out_features,
                dtype=self.weight.dtype,
                device=self.weight.device,
            )
            bias_init = init_rsqrt_uniform_(bias_init, self.in_features)
            with torch.inference_mode():
                self.bias.copy_(bias_init)

    def forward(self, x: Tensor) -> Tensor:
        # x.shape == (B, K, D)
        assert x.ndim == 3

        # >>> The equation (5) from the BatchEnsemble paper (arXiv v2).
        if self.r is not None:
            x = x * self.r
        x = x @ self.weight.T
        if self.s is not None:
            x = x * self.s
        # <<<

        if self.bias is not None:
            x = x + self.bias
        return x


def make_efficient_ensemble(module: nn.Module, **kwargs) -> None:
    """Replace torch.nn.Linear modules with LinearEfficientEnsemble.

    NOTE
    In the paper, there are no experiments with networks with normalization layers.
    Perhaps, their trainable weights (the affine transformations) also need
    "ensemblification" as in the paper about "FiLM-Ensemble".
    Additional experiments are required to make conclusions.
    """
    for name, submodule in list(module.named_children()):
        if isinstance(submodule, nn.Linear):
            module.add_module(
                name,
                LinearEfficientEnsemble(
                    in_features=submodule.in_features,
                    out_features=submodule.out_features,
                    bias=submodule.bias is not None,
                    **kwargs,
                ),
            )
        else:
            make_efficient_ensemble(submodule, **kwargs)


class MLP(nn.Module):
    def __init__(
            self,
            *,
            d_in: None | int = None,
            d_out: None | int = None,
            n_blocks: int,
            d_block: int,
            dropout: float,
            activation: str = 'ReLU',
    ) -> None:
        super().__init__()

        d_first = d_block if d_in is None else d_in
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_first if i == 0 else d_block, d_block),
                    getattr(nn, activation)(),
                    nn.Dropout(dropout),
                )
                for i in range(n_blocks)
            ]
        )
        self.output = None if d_out is None else nn.Linear(d_block, d_out)

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        if self.output is not None:
            x = self.output(x)
        return x


def get_experts_block(in_features: int, out_features: int, num_experts: int, activation: str,
                      dropout: None | bool, ) -> nn.ModuleList:
    w = torch.zeros(num_experts, in_features, out_features)

    w = init_rsqrt_uniform_(w, w.shape[-1])
    out = nn.ModuleList([nn.Parameter(w), getattr(nn, activation)()])
    if dropout is not None:
        out.append(nn.Dropout(dropout))
    return out


class BMoE(nn.Module):
    def __init__(
            self,
            *,
            d_in: None | int = None,
            d_out: None | int = None,
            n_blocks: int,
            d_block: int,
            dropout: float,
            activation: str = 'ReLU',
            num_experts: int,
            gating_type: str,  # ['standard' or 'bayesian']
            kl_factor: int,
            gating_prior_std: float,
    ) -> None:
        assert d_out is not None, "the output layer must be added to the MoE"
        assert gating_type in ['standard', 'bayesian']
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", device)
        self.device = device
        self.n_blocks = n_blocks
        self.kl_factor = kl_factor
        self.num_experts = num_experts
        self.gating_type = gating_type
        print(f'gating type:{self.gating_type}')
        d_first = d_block // num_experts if d_in is None else d_in

        self.stat_alpha_sum = None
        # Gating network
        self.gating_type = gating_type

        self.Weights = nn.ParameterList()
        for i in range(n_blocks + 1):  # one more for the output layer!
            w = torch.zeros(num_experts, d_first if i == 0 else d_block // num_experts,
                            d_out if i == n_blocks else d_block // num_experts)
            w = init_rsqrt_uniform_(w, w.shape[-1])
            self.Weights.append(nn.Parameter(w))

        self.activation = getattr(nn, activation)()

        self.dropout = nn.Dropout(dropout) if self.gating_type == 'standard' else None

        if self.gating_type == 'standard':
            self.gate = nn.Sequential(
                nn.Linear(d_first, num_experts),
                nn.Softmax(dim=-1)
            )

        elif self.gating_type == 'bayesian':
            self.gate = BayesianGatingNetwork(
                in_features=d_first,
                num_experts=num_experts,
                prior_std=gating_prior_std,
                device=self.device,
            )
        else:
            raise ValueError(f'The gating type "{self.gating_type}" is not supported.')

    def forward(self, x: Tensor, num_samples: int = 0, return_average: bool = True) -> Tensor:
        """
        If self.training is True:
           - Sample one alpha from gate (as usual),
           - Optionally store statistics,
           - Compute and return the weighted sum of expert outputs.
        If self.training is False (eval mode):
           - Sample 10 alphas from gate,
           - Compute expert outputs once (they're standard),
           - Average the weighted sums over those 10 alpha samples.
        """
        # print(f'num samples:{num_samples}')
        # TODO: improve code clarity
        if self.training or num_samples < 2 or self.gating_type == 'standard':

            # [batch_size, num_experts] -> [num_experts, batch_size]
            alpha = self.gate(x, num_samples=1) if self.gating_type == 'bayesian' \
                else self.gate(x)
            alpha = alpha.transpose(-1, -2)
        else:
            # [num_samples, batch_size, num_experts] -> [num_samples, num_experts, batch_size]
            alpha = self.gate(x, num_samples=num_samples).permute(0, 2, 1)

        for i in range(self.n_blocks + 1):
            x = torch.einsum('...nd,...dh->...nh', x, self.Weights[i])
            if i < self.n_blocks:
                x = self.activation(x)
                if self.dropout is not None:
                    x = self.dropout(x)

        if self.training or num_samples < 2 or self.gating_type == 'standard':
            output = torch.sum(alpha.unsqueeze(-1) * x, dim=0)
        else:
            # EVAL MODE (Bayesian ensemble)
            weighted_expert_outputs = alpha.unsqueeze(-1) * x.unsqueeze(0)

            # 4) Sum over experts => [10, batch_size, output_dim]
            weighted_sums = torch.sum(weighted_expert_outputs, dim=1)

            # [ num_samples, batch_size, output_dim]
            if return_average:
                output = weighted_sums.mean(dim=0)
            else:
                output = weighted_sums
        return output

    def get_kl_loss(self):
        """
        Only gating is Bayesian, so just return gating's KL.
        """
        if self.gating_type == 'standard':
            return torch.tensor(0.0).to(self.device)
        elif self.gating_type == 'bayesian':
            return self.kl_factor * self.gate.kl_loss()
        else:
            assert False, f'The gating type {self.gating_type} is not supported'


class MoIEBlock(nn.Module):
    """
    One "layer" containing K linear experts of shape (in_dim -> out_dim).
    In forward pass, we:
      1) Pass x through each expert -> Z^{(k)}  [size (B, out_dim)]
      2) Compute alpha * Z^{(k)} and sum across k -> Z_combined
      3) Optionally apply an activation (e.g. ReLU).
    """

    def __init__(self, in_dim, out_dim, num_experts, activation=True):
        super().__init__()
        self.num_experts = num_experts
        self.activation = activation

        # Create K linear experts as a ModuleList
        self.experts = nn.ModuleList([
            nn.Linear(in_dim, out_dim) for _ in range(num_experts)
        ])

        if self.activation:
            self.relu = nn.ReLU()

    def forward(self, x, alpha):
        """
        Args:
            x: shape (B, in_dim)
            alpha: shape (B, K) gating coefficients (one row per sample).
        Returns:
            combined: shape (B, out_dim)
        """
        B = x.size(0)  # batch size

        # 1) For each expert k, compute Z^{(k)} = x W^{(k)} + b^{(k)}.
        #    We'll stack them to shape (K, B, out_dim) for convenience.
        expert_outputs = []
        for k in range(self.num_experts):
            Z_k = self.experts[k](x)  # shape (B, out_dim)
            expert_outputs.append(Z_k)
        # Stack => shape (K, B, out_dim)
        expert_outputs = torch.stack(expert_outputs, dim=0)

        # 2) Combine with alpha: Z_combined(i,:) = sum_k alpha[i,k] * Z_k(i,:)
        #    We can do this in a batched manner:
        #    Make alpha shape (B, K, 1) => then broadcast multiply with
        #    expert_outputs (K, B, out_dim) after transposing or rearranging.
        #    Easiest is to transpose expert_outputs to (B, K, out_dim) first.
        expert_outputs = expert_outputs.transpose(0, 1)  # => (B, K, out_dim)

        # alpha: (B, K) => alpha.unsqueeze(-1): (B, K, 1)
        alpha_3d = alpha.unsqueeze(-1)  # => (B, K, 1)

        # Multiply elementwise and sum over K => (B, out_dim)
        combined = (expert_outputs * alpha_3d).sum(dim=1)  # (B, out_dim)

        # 3) Optional activation
        if self.activation:
            combined = self.relu(combined)
        return combined


class BMoIE(nn.Module):
    def __init__(
            self,
            *,
            d_in: None | int = None,
            d_out: None | int = None,
            n_blocks: int,
            d_block: int,
            dropout: float,
            activation: str = 'ReLU',
            # activation: str = 'GELU',
            num_experts: int,
            gating_type: str,  # ['standard' or 'bayesian']
            kl_factor: int,
            gating_prior_std: float,
            device: str = 'cpu',  # TODO: check whether is it necessary to pass
    ) -> None:
        assert gating_type in ['standard', 'bayesian']
        super().__init__()
        self.device = device
        self.num_experts = num_experts
        self.gating_type = gating_type
        self.kl_factor = kl_factor
        print(f'gating type: {self.gating_type}')
        d_first = d_block // num_experts if d_in is None else d_in

        self.stat_alpha_sum = None
        # Gating network
        self.gating_type = gating_type
        if self.gating_type == 'standard':
            self.gate = nn.Sequential(
                nn.Linear(d_first, num_experts),
                nn.Softmax(dim=-1)
            )

            self.blocks = nn.ModuleList(
                [
                    nn.Sequential(
                        MoIEBlock(d_first if i == 0 else d_block // num_experts, d_block // num_experts, num_experts,
                                  activation=False),
                        getattr(nn, activation)(),
                        nn.Dropout(dropout)
                    )
                    for i in range(n_blocks)
                ]
            )
        elif self.gating_type == 'bayesian':
            self.gate = BayesianGatingNetwork(
                in_features=d_first,
                num_experts=num_experts,
                prior_std=gating_prior_std,
                device=self.device,
            )

            self.blocks = nn.ModuleList(
                [
                    nn.Sequential(
                        MoIEBlock(d_first if i == 0 else d_block // num_experts, d_block // num_experts, num_experts,
                                  activation=False),
                        getattr(nn, activation)()
                    )
                    for i in range(n_blocks)
                ]
            )
        else:
            assert False, f'The gating type {self.gating_type} is not supported'

        self.output = None if d_out is None else MoIEBlock(d_block // num_experts, d_out, num_experts, activation=False)
        print(f'd_out:{d_out}')
        print(self.blocks)
        print('output:')
        print(self.output)

        self.device = device
        self.gating_type = gating_type
        self.stat_alpha_sum = None

    def forward(self, x):
        """
        x shape: (B, input_dim)
        alpha shape: (B, K)

        For simplicity, we'll assume we use the *same* alpha at each layer.
        If you want different gating per layer, you'd have multiple gating nets
        or a more advanced design.
        """
        alpha = self.gate(x)  # shape (B, K)

        # store for later analysis
        if self.training:
            if self.stat_alpha_sum is None:
                self.stat_alpha_sum = alpha.sum(axis=0).detach().cpu().numpy()
            else:
                self.stat_alpha_sum += alpha.sum(axis=0).detach().cpu().numpy()
        # if np.random.random() < 0.01:
        #     print(f'alphas:{self.stat_alpha_mean}')

        # Pass through 1st MoE block
        for block in self.blocks:
            x = block[0](x, alpha)  # Pass both arguments to the first block
            for i in range(1, len(block)):
                x = block[i](x)  # Apply the activation and dropout
        if self.output is not None:
            x = self.output(x, alpha)
        return x

    # expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
    #
    # # Weighted sum of expert outputs
    # output = torch.sum(alpha.unsqueeze(-1) * expert_outputs, dim=1)
    # return output

    def get_kl_loss(self):
        """
        Only gating is Bayesian, so just return gating's KL.
        """
        if self.gating_type == 'standard':
            return torch.tensor(0.0).to(self.device)
        elif self.gating_type == 'bayesian':
            return self.kl_factor * self.gate.kl_loss()
        else:
            assert False, f'The gating type {self.gating_type} is not supported'


_CUSTOM_MODULES = {
    # https://docs.python.org/3/library/stdtypes.html#definition.__name__
    CustomModule.__name__: CustomModule
    for CustomModule in [
        rtdl_num_embeddings.LinearEmbeddings,
        rtdl_num_embeddings.LinearReLUEmbeddings,
        rtdl_num_embeddings.PeriodicEmbeddings,
        PiecewiseLinearEmbeddings,
        MLP,
        BMoE,
        BMoIE
    ]
}


def make_module(type: str, *args, **kwargs) -> nn.Module:
    Module = getattr(nn, type, None)
    if Module is None:
        Module = _CUSTOM_MODULES[type]
    return Module(*args, **kwargs)


def get_n_parameters(m: nn.Module):
    return sum(x.numel() for x in m.parameters() if x.requires_grad)


@torch.inference_mode()
def compute_parameter_stats(module: nn.Module) -> dict[str, dict[str, float]]:
    stats = {'norm': {}, 'gradnorm': {}, 'gradratio': {}}
    for name, parameter in module.named_parameters():
        stats['norm'][name] = parameter.norm().item()
        if parameter.grad is not None:
            stats['gradnorm'][name] = parameter.grad.norm().item()
            # Avoid computing statistics for zero-initialized parameters.
            if (parameter.abs() > 1e-6).any():
                stats['gradratio'][name] = (
                    (parameter.grad.abs() / parameter.abs().clamp_min_(1e-6))
                    .mean()
                    .item()
                )
    stats['norm']['model'] = (
        torch.cat([x.flatten() for x in module.parameters()]).norm().item()
    )
    stats['gradnorm']['model'] = (
        torch.cat([x.grad.flatten() for x in module.parameters() if x.grad is not None])
        .norm()
        .item()
    )
    return stats


# ======================================================================================
# Optimization
# ======================================================================================
def default_zero_weight_decay_condition(
        module_name: str, module: nn.Module, parameter_name: str, parameter: Parameter
):
    from rtdl_num_embeddings import _Periodic

    del module_name, parameter
    return parameter_name.endswith('bias') or isinstance(
        module,
        nn.BatchNorm1d
        | nn.LayerNorm
        | nn.InstanceNorm1d
        | rtdl_revisiting_models.LinearEmbeddings
        | rtdl_num_embeddings.LinearEmbeddings
        | rtdl_num_embeddings.LinearReLUEmbeddings
        | _Periodic,
    )


def make_parameter_groups(
        module: nn.Module,
        zero_weight_decay_condition=default_zero_weight_decay_condition,
        custom_groups: None | list[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    if custom_groups is None:
        custom_groups = []
    custom_params = frozenset(
        itertools.chain.from_iterable(group['params'] for group in custom_groups)
    )
    assert len(custom_params) == sum(
        len(group['params']) for group in custom_groups
    ), 'Parameters in custom_groups must not intersect'
    zero_wd_params = frozenset(
        p
        for mn, m in module.named_modules()
        for pn, p in m.named_parameters()
        if p not in custom_params and zero_weight_decay_condition(mn, m, pn, p)
    )
    default_group = {
        'params': [
            p
            for p in module.parameters()
            if p not in custom_params and p not in zero_wd_params
        ]
    }
    return [
        default_group,
        {'params': list(zero_wd_params), 'weight_decay': 0.0},
        *custom_groups,
    ]


def make_optimizer(type: str, **kwargs) -> torch.optim.Optimizer:
    Optimizer = getattr(torch.optim, type)
    return Optimizer(**kwargs)


# ======================================================================================
# Training
# ======================================================================================
def zero_grad_forward_backward(
        optimizer: torch.optim.Optimizer,
        step_fn: Callable[[Tensor], Tensor],  # step_fn: batch_idx -> loss
        get_kl_loss: Callable[[], Tensor],  # zero for every models except Bayesian ones
        batch_idx: Tensor,
        chunk_size: int,
        grad_scaler: None | torch.cuda.amp.GradScaler = None,  # type: ignore[code]

) -> tuple[Tensor, Tensor, int]:
    """
    This is a standard training step. Additionally, it supports:

    - Training by chunks if the whole batch does not fit into GPU.
    - Gradient scaling for training in FP16.
    """
    backward = (
        Tensor.backward
        if grad_scaler is None
        else lambda x: grad_scaler.scale(x).backward()  # type: ignore[code]
    )
    batch_size = len(batch_idx)
    loss = None
    while chunk_size != 0:
        optimizer.zero_grad()

        try:
            if batch_size <= chunk_size:
                # The simple forward-backward.
                kl_loss = get_kl_loss()
                loss = step_fn(batch_idx) + kl_loss
                backward(loss)
            else:
                # Forward-backward by chunks.
                # Mathematically, this is equivalent to the simple forward-backward.
                # Technically, this implementations uses less memory.
                loss = None
                for chunk_idx in batch_idx.split(chunk_size):
                    chunk_loss = step_fn(chunk_idx)
                    kl_loss = get_kl_loss()
                    chunk_loss = (chunk_loss + kl_loss) * (len(chunk_idx) / batch_size)
                    backward(chunk_loss)
                    if loss is None:
                        loss = chunk_loss.detach()
                    else:
                        loss += chunk_loss.detach()
        except RuntimeError as err:
            if not is_oom_exception(err):
                raise
            delu.cuda.free_memory()
            chunk_size //= 2

        else:
            break

    if not chunk_size:
        raise RuntimeError('Not enough memory even for chunk_size=1')
    return cast(Tensor, loss), cast(Tensor, kl_loss), chunk_size
