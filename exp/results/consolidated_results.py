import os
import json
import pandas as pd
import numpy as np

# List to store data for the DataFrame
data_list = []

# Directories to search for report.json files
# directories_to_search = ["moe/", "bmoe/", "moe-piecewiselinear/", "bmoe-piecewiselinear/", "../mlp/",
#                          '../mlp-piecewiselinear/']
type = 'x2'  # or x1 or x2 or gumbel
if type == 'x1':
    directories_to_search = ["evaluation_15_04_2024/moe", "evaluation_15_04_2024/moe-piecewiselinear", "../mlp/",
                             '../mlp-piecewiselinear/']
    output_file = "consolidated_report.xlsx"
    main_model_type = 'MoE'
elif type == 'x2':
    directories_to_search = ["tuning_x2/moe-piecewiselinear", "tuning_x2/moe", "../mlp/",
                             '../mlp-piecewiselinear/',
                             "evaluation_15_04_2024/moe", "evaluation_15_04_2024/moe-piecewiselinear"]
    output_file = "consolidated_report_x2.xlsx"
    main_model_type = 'MoE_x2'
elif type == 'BMoE':
    directories_to_search = ["evaluation_15_04_2024/bmoe", "evaluation_15_04_2024/bmoe-piecewiselinear", "../mlp/",
                             '../mlp-piecewiselinear/',
                             "evaluation_15_04_2024/moe", "evaluation_15_04_2024/moe-piecewiselinear"]
    output_file = "consolidated_report_bmoe.xlsx"
    main_model_type = 'BMoE'
elif type == 'gumbel':
    directories_to_search = ["gumbel_tuning_results/bmoe", "gumbel_tuning_results/bmoe-piecewiselinear", "../mlp/",
                             '../mlp-piecewiselinear/',
                             "evaluation_15_04_2024/moe", "evaluation_15_04_2024/moe-piecewiselinear"]

    output_file = "consolidated_report_gumbel.xlsx"
    main_model_type = 'BMoE'
else:
    assert False, 'type is not supported'
# Recursively search for report.json files in '0-tuning' directories
for base_dir in directories_to_search:
    for root, _, files in os.walk(base_dir):
        if os.path.basename(root) == "0-tuning":  # Only process '0-tuning' directories
            for file in files:
                if file == "report.json":
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                        # Extract required data
                        gpus = data.get('gpus')
                        time = data.get('time')
                        backbone_type = data['best']['config']['model']['backbone']['type']
                        dataset_name = data['best']['config']['data']['path'].split('/')[-1]
                        n_parameters = data['best']['n_parameters']
                        n_blocks = data['best']['config']['model']['backbone']['n_blocks']
                        d_block = data['best']['config']['model']['backbone']['d_block']
                        dropout = data['best']['config']['model']['backbone']['dropout']
                        num_experts = data['best']['config']['model']['backbone'].get('num_experts', None)
                        if (num_experts is None) and (backbone_type == 'BMoE'):

                            num_experts = data['best']['config']['model']['backbone']['d_block'] // \
                                          data['best']['config']['model']['backbone']['d_block_per_expert']
                        else:
                            num_experts = 1
                        fc_size = d_block / num_experts if num_experts is not None else None
                        test_score = data['best']['metrics']['test']['score']
                        train_score = data['best']['metrics']['train']['score']
                        weighted_avg = data['best']['metrics']['test'].get('weighted avg', None)
                        # Determine task type
                        task = 'classification' if weighted_avg is not None else 'regression'

                        # Extract and modify backbone name

                        if (backbone_type == 'BMoE') and data['best']['config']['model']['backbone'][
                            'gating_type'] == 'standard':
                            backbone_type = 'MoE'
                        if 'config' in data and 'space' in data['config'] and \
                                'model' in data['config']['space'] and \
                                'num_embeddings' in data['config']['space']['model']:
                            backbone_type = f"E+{backbone_type}"

                        if 'tuning_x2' in base_dir:
                            backbone_type += '_x2'
                        # Append data as a dictionary to the list
                        data_list.append({
                            'dataset_name': dataset_name,
                            'backbone_type': backbone_type,
                            'gpus': gpus,
                            'time': time,
                            'n_parameters': n_parameters,
                            'n_blocks': n_blocks,
                            'num_experts': num_experts,
                            'dropout': dropout,
                            'd_block': d_block,
                            'fc_size': fc_size,
                            'test_score': test_score,
                            'train_score': train_score,
                            'weighted_avg': weighted_avg,
                            'task': task
                        })
                    except (KeyError, TypeError, json.JSONDecodeError) as e:
                        print(f"Error processing {file_path}: {e}")

# Create the DataFrame
df = pd.DataFrame(data_list)
# Set double index
# df = df.loc[df['dataset_name'].isin(["classif-num-medium-4-wine",
#                                      "classif-num-medium-3-wine",
#                                      "classif-num-medium-4-phoneme",
#                                      "classif-num-medium-2-wine",
#                                      "classif-num-medium-1-wine",
#                                      "classif-num-medium-1-phoneme",
#                                      "classif-num-medium-0-wine",
#                                      "classif-num-medium-0-phoneme",
#                                      "house",
#                                      "classif-num-medium-2-phoneme",
#                                      "adult",
#                                      "california",
#                                      "churn",
#                                      "classif-num-medium-3-phoneme"])]
# datasets = pd.read_excel('../../stat/datasets_info.xlsx')
# print(datasets.columns)
# df = pd.merge(df, datasets[['name', 'id']], left_on='dataset_name', right_on='name').drop(['name'], axis=1)
if main_model_type in ['BMoE', 'gumbel']:
    df = df.loc[df['backbone_type'].isin(['MLP', 'E+MLP', 'BMoE', 'E+BMoE', 'MoE', 'E+MoE'])]
    # df = df.loc[df['backbone_type'].isin(['E+MLP', 'E+BMoE', 'E+MoE'])]
else:
    df = df.loc[df['backbone_type'].isin(['MoE', 'E+MoE', 'MLP', 'E+MLP', 'MoE_x2', 'E+MoE_x2', 'BMoE', 'E+BMoE'])]

calculated_datasets = df.loc[df['backbone_type'] == main_model_type]['dataset_name'].values
calculated_datasets_emoe = df.loc[df['backbone_type'] == 'E+' + main_model_type]['dataset_name'].values
print(f'MoE - E+MoE datasets:{np.setdiff1d(calculated_datasets, calculated_datasets_emoe)}')
print(f'E+MoE - MoE datasets:{np.setdiff1d(calculated_datasets_emoe, calculated_datasets)}')

all_datasets = df.loc[df['backbone_type'] == 'E+MLP']['dataset_name'].values
print('missing datasets:')
print(np.setdiff1d(all_datasets, calculated_datasets))
df = df.loc[df['dataset_name'].isin(calculated_datasets_emoe)]
df['dataset_index'] = df['dataset_name'].apply(lambda x: x.split('-')[-1])
df['time'] = pd.to_timedelta(df['time'])

# df = df.groupby(['dataset_index', 'backbone_type']).agg(
#     {'dataset_name': list,
#      'gpus': list, 'time': list, 'n_parameters': lambda x: int(np.mean(x)),
#      'test_score': np.mean, 'task': lambda x: np.unique(x)[0], 'n_blocks': [list, 'max']}).assign(
#     rank=lambda x: x.groupby('dataset_index')['test_score'].rank(ascending=False, method='min')).reset_index()


df = df.groupby(['dataset_index', 'backbone_type']).agg(
    dataset_name=('dataset_name', list),
    gpus=('gpus', list),
    time=('time', 'sum'),
    n_parameters=('n_parameters', lambda x: int(np.mean(x))),
    test_score=('test_score', 'mean'),
    train_score=('train_score', 'mean'),
    task=('task', lambda x: np.unique(x)[0]),
    n_blocks_list=('n_blocks', list),
    n_blocks_max=('n_blocks', 'max'),

    num_experts_list=('num_experts', list),
    num_experts_max=('num_experts', 'max'),

    d_block_list=('d_block', list),
    d_block_max=('d_block', 'max'),

    fc_size_list=('fc_size', list),
    fc_size_median=('fc_size', 'median'),

    dropout_list=('dropout', list),
    dropout_median=('dropout', 'median'),
).assign(
    rank=lambda x: x.groupby('dataset_index')['test_score'].rank(ascending=False, method='min'),
    train_rank=lambda x: x.groupby('dataset_index')['train_score'].rank(ascending=False, method='min')).reset_index()
# df.set_index(['dataset_name', 'backbone_type'], inplace=True)
print(df.shape)
# Save to Excel file

# Determine the backbone_type with rank 1 for each dataset_index
winners = df[df['rank'] == 1].groupby('dataset_index')['backbone_type'].first()
train_winners = df[df['train_rank'] == 1].groupby('dataset_index')['backbone_type'].first()

# Add the backbone_type_winner column
df['backbone_type_winner'] = df['dataset_index'].map(winners)
df['train_backbone_type_winner'] = df['dataset_index'].map(train_winners)
df['time'] = df['time'].apply(lambda x: str(x).split(" ")[-1])
df.to_excel(output_file)
print(f"Data has been saved to {output_file}")

# Calculate the average rank for each backbone_type
avg_rank = df.groupby('backbone_type')['rank'].mean()
avg_rank_train = df.groupby('backbone_type')['train_rank'].mean()

avg_n_param = df.groupby('backbone_type')['n_parameters'].mean()
print('average number of parameters:')
print(avg_n_param)

avg_n_param_clf = df.loc[df['task'] == 'classification'].groupby('backbone_type')['n_parameters'].mean()
print('average number of parameters for clf:')
print(avg_n_param_clf)

avg_n_param_reg = df.loc[df['task'] == 'regression'].groupby('backbone_type')['n_parameters'].mean()
print('average number of parameters for reg:')
print(avg_n_param_reg)

avg_rank_clf = df.loc[df['task'] == 'classification'].groupby('backbone_type')['rank'].mean()
avg_rank_reg = df.loc[df['task'] == 'regression'].groupby('backbone_type')['rank'].mean()



print('avg rank train:')
print(avg_rank_train)
print('avg rank for clf')
print(avg_rank_clf)

print('avg rank for reg')
print(avg_rank_reg)

print(len(df.loc[df['backbone_type'] == 'MoE']))
print(len(df.loc[df['backbone_type'] == 'E+MoE']))
print(len(df.loc[df['backbone_type'] == 'MLP']))
print(len(df.loc[df['backbone_type'] == 'E+MLP']))
print(len(df.loc[df['backbone_type'] == 'E+MoE_x2']))
print(len(df.loc[df['backbone_type'] == 'MoE_x2']))


# Print the result
print('avg rank:')
print(avg_rank)