import os
import tomli
import tomli_w
from pathlib import Path


def modify_and_copy_configs(src_folder, dest_folder, model_type: str):
    assert model_type in ['moe', 'bmoe', 'deepbmoe', 'gmlp_bmoe', 'bmoe_adapter',
                          'bmoe_adapter_sigmoid', 'bmoe_adapter_sigmoid_attention',
                          'bmoe_adapter_sigmoid_big',
                          'bmoe_adapter_sigmoid_medium',
                          'bmoe_adapter_sigmoid_normal_init',
                          'bmoe_adapter_sigmoid_kmeans',
                          'bmoe_adapter_sigmoid_tabm_mini', f'bmoe_adapter_sigmoid_tabm_mini_top_{top_k}',
                          f'bmoe_adapter_sigmoid_tabm_mini_top_{top_k}_64', 'gg_bmoe_tabm_mini',
                          f'gg_bmoe_tabm_mini_{hidden_dim}'], "Incorrect model_type"
    # Ensure destination folder exists
    Path(dest_folder).mkdir(parents=True, exist_ok=True)

    # Iterate through all subdirectories in the source folder
    for dataset_name in os.listdir(src_folder):
        dataset_path = os.path.join(src_folder, dataset_name)
        if not os.path.isdir(dataset_path):
            continue

        src_config_path = os.path.join(dataset_path, "0-tuning.toml")
        if not os.path.exists(src_config_path):
            print(f"Config not found: {src_config_path}")
            continue

        # Load the original config
        with open(src_config_path, "rb") as f:
            config = tomli.load(f)

        # Modify the config
        config["function"] = "_bin.model.main"

        # Modify optimizer settings
        optimizer = config["space"]["optimizer"]
        if model_type not in ['gg_bmoe_tabm_mini', f'gg_bmoe_tabm_mini_{hidden_dim}']:
            if model_type in ('bmoe_adapter_sigmoid_big', 'bmoe_adapter_sigmoid_medium', 'bmoe_adapter_sigmoid_attention'):
                optimizer["lr"] = ["_tune_", "loguniform", 0.0001, 0.005]
            elif model_type not in ['bmoe_adapter_sigmoid_tabm_mini', f'bmoe_adapter_sigmoid_tabm_mini_top_{top_k}',
                                    f'bmoe_adapter_sigmoid_tabm_mini_top_{top_k}_64']:
                optimizer["lr"] = ["_tune_", "loguniform", 0.0003, 0.01]
            elif model_type == f'bmoe_adapter_sigmoid_tabm_mini_top_{top_k}_64':
                optimizer["lr"] = ["_tune_", "loguniform", 0.0001, 0.01]
        # if model_type == 'bmoe':
        #     optimizer['type'] = 'Adam'
        #     del optimizer['weight_decay']
        # Modify model backbone
        model_backbone = config["space"]["model"]["backbone"]
        if model_type == 'deepbmoe':
            model_backbone['type'] = 'DeepBMoE'
        else:
            model_backbone["type"] = "BMoE"
        # model_backbone["n_blocks"] = ["_tune_", "int", 1, 2]

        # model_backbone["d_block"] = ["_tune_", "int", 128, 2048, 32]
        if model_type == 'bmoe_adapter_sigmoid_big':
            model_backbone["n_blocks"] = ["_tune_", "int", 1, 4]
            model_backbone["d_block"] = ["_tune_", "int", 512, 4096, 128]
            model_backbone["d_block_per_expert"] = ["_tune_", "int", 128, 256, 128]
        elif model_type == 'bmoe_adapter_sigmoid_medium':
            model_backbone["n_blocks"] = ["_tune_", "int", 1, 4]
            model_backbone["d_block"] = ["_tune_", "int", 512, 4096, 128]
            model_backbone["d_block_per_expert"] = ["_tune_", "int", 64, 128, 64]
        elif model_type == f'bmoe_adapter_sigmoid_tabm_mini_top_{top_k}_64':
            model_backbone["d_block"] = ["_tune_", "int", 1280, 2560, 64]
            model_backbone["d_block_per_expert"] = 64
        elif model_type == f'gg_bmoe_tabm_mini_{hidden_dim}':
            model_backbone["d_block"] = ["_tune_", "int", 4*hidden_dim, 32*hidden_dim, 2*hidden_dim]
            model_backbone["d_block_per_expert"] = hidden_dim
        else:
            model_backbone["d_block"] = ["_tune_", "int", 128, 1280, 64]
            model_backbone["d_block_per_expert"] = ["_tune_", "int", 32, 64, 32]
        # model_backbone["d_block"] = ["_tune_", "int", 256, 2560, 128]
        # model_backbone["d_block_per_expert"] = ["_tune_", "int", 32, 128, 32]

        if model_type in ('bmoe', 'deepbmoe', 'gmlp_bmoe', 'bmoe_adapter', 'gg_bmoe_tabm_mini', f'gg_bmoe_tabm_mini_{hidden_dim}'):
            # model_backbone["dropout"] = 0.0
            # model_backbone["gating_prior_std"] = ["_tune_", "uniform", 0.1, 1.0]
            # model_backbone["kl_factor"] = ["_tune_", "loguniform", 0.001, 1.0]
            model_backbone["default_num_samples"] = 10
            model_backbone["tau"] = ["_tune_", "uniform", 0.5, 3.0]
        elif model_type not in ('bmoe_adapter_sigmoid', 'bmoe_adapter_sigmoid_attention',
                                'bmoe_adapter_sigmoid_normal_init',
                                'bmoe_adapter_sigmoid_big', 'bmoe_adapter_sigmoid_medium',
                                'bmoe_adapter_sigmoid_kmeans',
                                'bmoe_adapter_sigmoid_tabm_mini',
                                f'bmoe_adapter_sigmoid_tabm_mini_top_{top_k}',
                                f'bmoe_adapter_sigmoid_tabm_mini_top_{top_k}_64'):
            # model_backbone["num_experts"] = ["_tune_", "int", 4, 40, 4]
            model_backbone["gating_prior_std"] = 1.0
            model_backbone["kl_factor"] = 1e-2

        if model_type == 'gmlp_bmoe':
            model_backbone["expert_type"] = 'gMLP'

        if model_type in ['bmoe_adapter', 'bmoe_adapter_sigmoid', 'bmoe_adapter_sigmoid_attention',
                          'bmoe_adapter_sigmoid_normal_init',
                          'bmoe_adapter_sigmoid_big',
                          'bmoe_adapter_sigmoid_medium',
                          'bmoe_adapter_sigmoid_kmeans', 'bmoe_adapter_sigmoid_tabm_mini',
                          f'bmoe_adapter_sigmoid_tabm_mini_top_{top_k}',
                          f'bmoe_adapter_sigmoid_tabm_mini_top_{top_k}_64']:
            model_backbone["adapter"] = True
        # Add new variables
        if model_type in ('bmoe', 'deepbmoe', 'gmlp_bmoe', 'bmoe_adapter', 'gg_bmoe_tabm_mini', f'gg_bmoe_tabm_mini_{hidden_dim}'):
            model_backbone["gating_type"] = "bayesian"
        elif model_type in ('bmoe_adapter_sigmoid', 'bmoe_adapter_sigmoid_normal_init',
                            'bmoe_adapter_sigmoid_big',
                            'bmoe_adapter_sigmoid_medium', 'bmoe_adapter_sigmoid_tabm_mini',
                            f'bmoe_adapter_sigmoid_tabm_mini_top_{top_k}',
                            f'bmoe_adapter_sigmoid_tabm_mini_top_{top_k}_64'):
            model_backbone["gating_type"] = "sigmoid_adapter"
        elif model_type == 'bmoe_adapter_sigmoid_kmeans':
            model_backbone["gating_type"] = "sigmoid_adapter_kmeans"
        elif model_type == 'bmoe_adapter_sigmoid_attention':
            model_backbone["gating_type"] = "sigmoid_adapter_attention"
            model_backbone["q_dim"] = 32
        else:
            model_backbone["gating_type"] = "standard"

        if model_type == 'bmoe_adapter_sigmoid_normal_init':
            model_backbone['adapter_init'] = 'normal'
        if model_type in [f'bmoe_adapter_sigmoid_tabm_mini_top_{top_k}', f'bmoe_adapter_sigmoid_tabm_mini_top_{top_k}_64']:
            model_backbone['top_k'] = top_k
        if model_type == f'bmoe_adapter_sigmoid_tabm_mini_top_{top_k}_64':
            config["space"]["model"]["k"] = 64
        # Define destination path and save the modified config
        dest_dataset_path = os.path.join(dest_folder, dataset_name)
        Path(dest_dataset_path).mkdir(parents=True, exist_ok=True)
        dest_config_path = os.path.join(dest_dataset_path, "0-tuning.toml")

        with open(dest_config_path, "wb") as f:
            tomli_w.dump(config, f)

        print(f"Modified config saved to: {dest_config_path}")

top_k = 32
hidden_dim = 128
# model_type = 'gmlp_bmoe'
# model_type = 'bmoe_adapter_sigmoid_normal_init'
# model_type = 'bmoe_adapter_sigmoid_attention'
# model_type = f'bmoe_adapter_sigmoid_tabm_mini_top_{top_k}_64'
model_type = f'gg_bmoe_tabm_mini_{hidden_dim}'
embeddings = "-piecewiselinear"
# embeddings = ""
# Define source and destination folders
# source_folder = f"exp/mlp{embeddings}"
source_folder = f"exp/tabm-mini{embeddings}/"
# source_folder = f"exp/mlp/why"
destination_folder = f"exp_advanced/{model_type}{embeddings}/"

# Run the function
modify_and_copy_configs(source_folder, destination_folder, model_type)
