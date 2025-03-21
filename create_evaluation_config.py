import json
from pathlib import Path
import sys

import os
import glob


def find_matching_files(target_folder, pattern="*/0-tuning/report.json"):
    # Use glob to search recursively for the pattern
    matching_files = glob.glob(os.path.join(target_folder, "**", pattern), recursive=True)
    return matching_files


# Example usage
target_folder = "/path/to/your/target/folder"  # Replace with your target folder path
files = find_matching_files(target_folder)

# Print the list of matching files
for file in files:
    print(file)
if __name__ == '__main__':
    _cwd = Path.cwd()
    assert _cwd.joinpath(
        '.git'
    ).exists(), 'The script must be run from the root of the repository'
    sys.path.append(str(_cwd))
    del _cwd

from lib import dump_config


def create_config(json_file: str, output: str, gating_type: str, model_type: str):
    # Create the directories if they don't exist
    os.makedirs(os.path.dirname(output), exist_ok=True)
    # Read the JSON file
    with open(json_file, "r") as f:
        data = json.load(f)

    config = data['best']['config']

    if model_type in ['mlp', 'gg_moe_no_sampling', 'tabm_mini']:
        dump_config(output, config)
        print(f"Converted {json_file} to {output} successfully!")
        return

    if model_type == 'tabm_mini_bmoe_adapter_sigmoid-piecewiselinear':
        config['model']['arch_type'] = 'tabm-mini'
        config['model']['k'] = 32
        # arch_type = "tabm-mini"
        # k = 32
        dump_config(output, config)
        print(f"Converted {json_file} to {output} successfully!")
        return

    if 'device' in config['model']['backbone']:
        config['model']['backbone'].pop('device')

    # config['model']['backbone']['num_experts'] = (config['model']['backbone']['d_block'] // hidden_size_per_expert) + 1
    # config['model']['backbone']['d_block'] = config['model']['backbone']['num_experts'] * hidden_size_per_expert

    config['model']['backbone']['type'] = 'BMoE'
    # config['model']['backbone']['gating_prior_std'] = 0.1
    # config['model']['backbone']['kl_factor'] = 1e-2
    if gating_type == 'bayesian':
        # config['model']['backbone']['dropout'] = 0.0
        config['n_bayesian_ensembles'] = [1, 5, 10, 100]
        config['num_samples'] = 100
        config['model']['backbone']['gating_type'] = 'bayesian'
        config['return_average'] = False
        config['model']['backbone']["default_num_samples"] = 10
    else:
        config['model']['backbone']['gating_type'] = 'standard'
    dump_config(output, config, force=True)

    print(f"Converted {json_file} to {output} successfully!")


def find_matching_files(target_folder, pattern="*/0-tuning/report.json"):
    # Use glob to search recursively for the pattern
    matching_files = glob.glob(os.path.join(target_folder, "**", pattern), recursive=True)
    return matching_files


if __name__ == '__main__':
    # Input and output file paths

    # model_type = 'mlp'
    model_type = 'gg_moe_no_sampling'
    # model_type = 'tabm_mini'
    # model_type = 'moe_adapter_sigmoid'
    # model_type = 'tabm_mini_bmoe_adapter_sigmoid-piecewiselinear'
    # input_folder = 'exp/results/evaluation_15_04_2024/moe-piecewiselinear'
    # input_folder = 'exp/results/gumbel_tuning_results/bmoe-piecewiselinear'
    # input_folder = 'exp/results/gumbel_tuning_results/bmoe-piecewiselinear'
    # input_folder = 'exp_advanced/results/bmoe_adapter_sigmoid-piecewiselinear/'
    # input_folder = 'exp/mlp-piecewiselinear/'

    # input_folder = 'exp/tabm-mini-piecewiselinear/'
    # input_folder = 'exp_advanced/results/bmoe_adapter_sigmoid_tabm_mini-piecewiselinear/'
    input_folder = 'exp_advanced/results/bmoe_adapter_sigmoid-piecewiselinear/'


    # output_folder = "exp/results/evaluation_16_04_2024/moe-piecewiselinear/"
    # output_folder = "exp_advanced/evaluation_10_03_2025/tabm-mini-piecewiselinear/"
    output_folder = "exp_advanced/evaluation_10_03_2025/bmoe_adapter_sigmoid_piecewiselinear/"
    # model_type = 'moe'

    # input_folder = 'exp/results/gumbel_tuning_results/bmoe-piecewiselinear/'
    # output_folder = "exp/results/gumbel_evaluation_configs/bmoe-piecewiselinear/"
    # model_type = 'bmoe'
    gating_type = 'standard' if model_type == 'moe' else 'bayesian'
    if model_type == 'tabm_mini_bmoe_adapter_sigmoid-piecewiselinear':
        gating_type = 'sigmoid_adapter'
    reports = find_matching_files(input_folder)
    for r in reports:
        output = output_folder + r.removeprefix(input_folder).removesuffix('0-tuning/report.json')
        output += '0-evaluation/0.toml'
        print(f'output:{output}')
        # Extract the directory path (excluding the file name)
        dir_path = os.path.dirname(output)
        create_config(r, output, gating_type, model_type)
