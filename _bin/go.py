import argparse
import sys
from pathlib import Path
from typing import cast

if __name__ == '__main__':
    assert (
        Path.cwd().joinpath('.git').exists()
    ), 'The script must be run from the root of the repository'
    sys.path.append(str(Path.cwd()))

import _bin.ensemble
import _bin.evaluate
import _bin.tune
import lib


def main(
    path: str | Path,  # "a/b/c/0-tuning" OR "a/b/c/0-bayesian-evaluation"
    function: None | str = None,
    n_seeds: int = _bin.evaluate.DEFAULT_N_SEEDS,
    n_ensembles: int = _bin.ensemble.DEFAULT_N_ENSEMBLES,
    ensemble_size: int = _bin.ensemble.DEFAULT_ENSEMBLE_SIZE,
    *,
    continue_: bool = False,
    force: bool = False,
):
    path = Path(path).resolve()
    if path.name.endswith(('-tuning', '-tuning.toml')):
        assert function is None
        tuning_output = path.with_suffix('')
        tuning_config = cast(_bin.tune.Config, lib.load_config(tuning_output))
        _bin.tune.main(tuning_config, tuning_output, continue_=continue_, force=force)
        evaluation_input = tuning_output
        evaluation_dir = tuning_output.with_name(
            tuning_output.name.replace('tuning', 'evaluation')
        )

    elif path.name.endswith('-evaluation'):
        assert function is not None
        evaluation_input = path
        evaluation_dir = path

    else:
        raise ValueError(f'Bad input path: {path}')

    _bin.evaluate.main(evaluation_input, n_seeds, function, force=force)
    # _bin.ensemble.main(evaluation_dir, n_ensembles, ensemble_size, force=force)


if __name__ == '__main__':
    lib.configure_libraries()

    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('--n_seeds', type=int, default=_bin.evaluate.DEFAULT_N_SEEDS)
    parser.add_argument('--function')
    parser.add_argument(
        '--n_ensembles', type=int, default=_bin.ensemble.DEFAULT_N_ENSEMBLES
    )
    parser.add_argument(
        '--ensemble_size', type=int, default=_bin.ensemble.DEFAULT_ENSEMBLE_SIZE
    )
    parser.add_argument('--continue', action='store_true', dest='continue_')
    parser.add_argument('--force', action='store_true')
    main(**vars(parser.parse_args(sys.argv[1:])))
