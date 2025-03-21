import os
import json
import pandas as pd

bayesian_ensemble_file_name = 'report_bayes_ensemble.json'
n_bayesian_ensembles = [1, 5, 10, 100]
import numpy as np

# type = 'gumbel'
type = 'advanced'


def find_report_files(base_path):
    """
    Find all `report.json` files matching the pattern */0-evaluation/{n_seed}/report.json.
    """
    report_files = []
    for root, _, files in os.walk(base_path):
        for file in files:
            if (file == "report.json" and "0-evaluation" in root) or (
                    file == "report.json" and "2-evaluation" in root):
                # if file == "report.json" and "0-evaluation" in root:
                report_files.append(os.path.join(root, file))
    return report_files


def extract_scores(report_files, n=0):
    """
    Extract scores from `report.json` files and group them by dataset_name.
    """
    data_scores = {}
    for file_path in report_files:
        if n != 0:
            dir_path = os.path.dirname(file_path)
            be_file_name = os.path.join(dir_path, bayesian_ensemble_file_name)
            if os.path.exists(be_file_name):
                with open(be_file_name, 'r') as f:
                    report_be = json.load(f)

        with open(file_path, 'r') as f:
            report = json.load(f)
            dataset_path = report['config']['data']['path'].removeprefix('data/')
            scores = {
                "train_score": report['metrics']['train']['score'] if 'train' in report['metrics'] else -999999,
                "val_score": report['metrics']['val']['score'],
                "test_score": report['metrics']['test']['score'] if n == 0 else report_be['metrics'][str(n)]['test'][
                    'score'],
                "n_parameters": report['n_parameters'] if 'n_parameters' in report else 0.0,
                "time": pd.to_timedelta(report['time'] if 'time' in report else 0)
            }
            # for n in n_bayesian_ensembles:
            #     scores['test_score_' + str(n)] = report['metrics']['test']['score']
            # dir_path = os.path.dirname(file_path)
            # be_file_name = os.path.join(dir_path, bayesian_ensemble_file_name)
            # if os.path.exists(be_file_name):
            #     with open(be_file_name, 'r') as f:
            #         report_be = json.load(f)
            #     for n in n_bayesian_ensembles:
            #         scores['test_score_' + str(n)] = report_be['metrics'][str(n)]['test']['score']
            #     # scores['test_score'] = scores['test_score_100']

            is_gbdt = True
            if 'gbdt' not in file_path:
                is_gbdt = False
                model_type = report['config']['model']['backbone']['type']

            # gbdt
            if 'catboost' in file_path:
                model_type = 'catboost'
            if 'xgboost' in file_path:
                model_type = 'xgboost'
            if 'lightgbm' in file_path:
                model_type = 'lightgbm'

            if (model_type == 'BMoE') and report['config']['model']['backbone']['gating_type'] == 'standard':
                model_type = 'MoE'
            if 'evaluation_results_x2' in file_path:
                model_type += '_x2'
            if 'gumbel_evaluation_results' in file_path:
                model_type += '_gumbel'
            if 'tabm-mini' in file_path:
                model_type += '_tabm-mini'
            elif 'exp/tabm' in file_path:
                model_type += '_tabm'
            if 'top_8_64' in file_path:
                model_type += '_top_8_64'

            if 'bmoe_adapter-piecewiselinear' in file_path:
                model_type += '_adapter'
            if 'bmoe_adapter_sigmoid-piecewiselinear' in file_path:
                model_type += '_adapter_sigmoid'
            if 'bmoe_adapter_sigmoid_tabm_mini' in file_path:
                model_type += '_adapter_sigmoid_tabm_mini'
            if 'bmoe_adapter_sigmoid_kmeans-piecewiselinear' in file_path:
                model_type += '_adapter_sigmoid_kmeans'
            if 'gmlp_bmoe-piecewiselinear' in file_path:
                model_type += '_gmlp'
            if 'bmoe_adapter_sigmoid_big-piecewiselinear' in file_path:
                model_type += '_big'
            if 'bmoe_adapter_sigmoid_medium-piecewiselinear' in file_path:
                model_type += '_medium'
            if 'bmoe_adapter_sigmoid_normal_init-piecewiselinear' in file_path:
                model_type += '_normal_init'

            if ('config' in report and 'model' in report['config'] and \
                'num_embeddings' in report['config']['model']) or \
                    (type == 'advanced' and 'Mercedes_Benz_Greener_Manufacturing' in file_path and not is_gbdt):
                model_type = f"E+{model_type}"

            scores['model_type'] = model_type if n == 0 else model_type + '_' + str(n)
            scores['gpus'] = report['gpus']

            scores['n_blocks'] = report['config']['model']['backbone']['n_blocks'] if not is_gbdt else 0.0
            scores['d_block'] = report['config']['model']['backbone']['d_block'] if not is_gbdt else 0.0
            scores['dropout'] = report['config']['model']['backbone']['dropout'] if not is_gbdt else 0.0

            scores['d_block_per_expert'] = report['config']['model']['backbone'].get(
                'd_block_per_expert') if not is_gbdt else 0.0
            scores['tau'] = report['config']['model']['backbone'].get('tau') if not is_gbdt else 0.0

            scores['lr'] = report['config']['optimizer']['lr'] if not is_gbdt else 0.0
            scores['weight_decay'] = report['config']['optimizer']['weight_decay'] if not is_gbdt else 0.0
            if dataset_path not in data_scores:
                data_scores[dataset_path] = {"train_score": [], "val_score": [], "test_score": [],
                                             'n_parameters': [], 'model_type': [], 'time': [], 'gpus': [], 'tau': [],
                                             'n_blocks': [], 'd_block': [], 'dropout': [], 'd_block_per_expert': [],
                                             'lr': [], 'weight_decay': []}
                # for n in n_bayesian_ensembles:
                #     data_scores[dataset_path]['test_score_' + str(n)] = []
            for key in scores:
                data_scores[dataset_path][key].append(scores[key])

    return data_scores


def calculate_statistics(data_scores):
    """
    Calculate mean and standard deviation of scores for each dataset_name.
    """
    records = []
    for dataset_path, scores in data_scores.items():
        record = {"dataset_name": dataset_path, 'model_type': scores['model_type'][0], 'gpus': scores['gpus'][0],
                  'n_blocks': scores['n_blocks'][0], 'd_block': scores['d_block'][0],
                  'd_block_per_expert': scores['d_block_per_expert'][0], 'dropout': scores['dropout'][0],
                  'lr': scores['lr'][0], 'weight_decay': scores['weight_decay'][0], 'tau': scores['tau'][0]}
        # be_test = ['test_score_' + str(n) for n in n_bayesian_ensembles]
        for metric in ["train_score", "val_score", "test_score", 'n_parameters', 'time']:  # + be_test:
            if metric == 'time':
                time_deltas_index = pd.TimedeltaIndex(scores[metric])
                mean_score = time_deltas_index.mean()
                std_score = time_deltas_index.std(ddof=1)
            else:
                mean_score = sum(scores[metric]) / len(scores[metric]) if scores[metric] else 0
                std_score = (
                    (sum((x - mean_score) ** 2 for x in scores[metric]) / (len(scores[metric]) - 1)) ** 0.5
                    if scores[metric]
                    else 0
                )
            record[f"{metric}_mean"] = mean_score
            record[f"{metric}_std"] = std_score
        records.append(record)
    return pd.DataFrame(records)


def save_to_excel(df, output_path, file_name):
    """
    Save the DataFrame to an Excel file.
    """
    output_file = os.path.join(output_path, file_name)
    df.to_excel(output_file, index=False)
    return output_file


def rank_models(dataframe):
    """
    Assign ranks to models based on the ranking algorithm provided.

    Args:
        dataframe (pd.DataFrame): Input dataframe with columns
                                  'dataset_index', 'test_score_mean', 'test_score_std'.

    Returns:
        pd.DataFrame: Dataframe with an additional 'rank' column for each dataset_index.
    """
    # Initialize an empty list to store ranked results
    ranked_dataframes = []

    # Iterate over each unique dataset_index
    for dataset_index, group in dataframe.groupby('dataset_index'):
        # Sort the group by test_score_mean in descending order
        group = group.sort_values('test_score_mean', ascending=False).reset_index(drop=True)
        group['rank'] = 0  # Initialize the rank column

        current_rank = 1  # Start with rank 1
        ref_index = 0  # Reference model index

        while ref_index < len(group):
            ref_mean = group.loc[ref_index, 'test_score_mean']
            ref_std = group.loc[ref_index, 'test_score_std']

            # Find models equal to the reference model according to the rule
            equal_models = (ref_mean - group['test_score_mean'] <= ref_std)
            group.loc[equal_models & (group['rank'] == 0), 'rank'] = current_rank

            # Move to the next model that is not equal to the reference model
            ref_index = group[group['rank'] == 0].index.min()

            # Increment rank if we move to a new reference model
            current_rank += 1

        ranked_dataframes.append(group)

    # Concatenate all ranked groups back together
    ranked_df = pd.concat(ranked_dataframes, ignore_index=True)
    return ranked_df


# Main execution
def main():
    # base_paths = ["exp/mlp", 'exp/results/evaluation_15_04_2024/moe_from_mlp',
    #               'exp/results/evaluation_15_04_2024/bmoe_from_mlp']  # Replace with your dataset folder path
    # base_paths = ["exp/mlp", 'exp/results/evaluation_15_04_2024/bmoe_from_mlp',]  # Replace with your dataset folder path
    # base_paths = ["exp/results/evaluation_15_04_2024/moe_from_mlp", 'exp/results/evaluation_15_04_2024/bmoe_from_mlp_std_01',]  # Replace with your dataset folder path
    if type == 'x2':
        base_paths = [

            "exp/results/evaluation_results_16_04_2024/moe",
            "exp/results/evaluation_results_x2/moe",
            # "exp/results/evaluation_results_16_04_2024/bmoe",
            "exp/results/evaluation_results_16_04_2024/moe-piecewiselinear",
            "exp/results/evaluation_results_x2/moe-piecewiselinear",
            # "exp/results/evaluation_results_16_04_2024/bmoe-piecewiselinear",
            # "exp/mlp",
            "exp/mlp-piecewiselinear"
        ]
    elif type == 'gumbel':
        base_paths = [

            "exp/results/evaluation_results_16_04_2024/moe",
            "exp/results/gumbel_evaluation_results/bmoe",
            # "exp/results/evaluation_results_16_04_2024/bmoe",
            "exp/results/evaluation_results_16_04_2024/moe-piecewiselinear",
            "exp/results/gumbel_evaluation_results/bmoe-piecewiselinear",
            # "exp/results/evaluation_results_16_04_2024/bmoe-piecewiselinear",
            "exp/mlp",
            "exp/mlp-piecewiselinear",
            # "exp/tabm-mini-piecewiselinear",
            # "exp/tabm-mini",
            # "exp/results/evaluation_results_x2/moe-piecewiselinear",
            # "exp/results/evaluation_results_x2/moe",
        ]
    elif type == 'advanced':
        base_paths = [
            "exp/mlp-piecewiselinear",
            # "exp/results/evaluation_results_16_04_2024/moe-piecewiselinear",
            "exp/results/gumbel_evaluation_results/bmoe-piecewiselinear",

            "exp_advanced/results/bmoe_adapter-piecewiselinear",
            "exp_advanced/results/bmoe_adapter_sigmoid-piecewiselinear",

            # "exp_advanced/results/bmoe_adapter_sigmoid_tabm_mini-piecewiselinear",
            # "exp_advanced/results/bmoe_adapter_sigmoid_tabm_mini_top_8_64-piecewiselinear",
            # "exp_advanced/results/bmoe_adapter_sigmoid_kmeans-piecewiselinear",
            # "exp_advanced/results/deepbmoe-piecewiselinear",
            # "exp_advanced/results/gmlp_bmoe-piecewiselinear",

            # "exp/tabm-mini-piecewiselinear",
            # "exp/tabm-piecewiselinear",

            # "exp_advanced/results/bmoe_adapter_sigmoid_big-piecewiselinear",
            # "exp_advanced/results/bmoe_adapter_sigmoid_medium-piecewiselinear",
            # "exp_advanced/results/bmoe_adapter_sigmoid_normal_init-piecewiselinear",

            # "exp_advanced/results/gbdt/catboost_",
            # "exp_advanced/results/gbdt/xgboost_",
            # "exp_advanced/results/gbdt/lightgbm_",
        ]
    else:
        base_paths = [
            "exp/results/evaluation_results_16_04_2024/moe",
            # "exp/results/evaluation_results_16_04_2024/bmoe",
            "exp/results/evaluation_results_16_04_2024/moe-piecewiselinear",
            # "exp/results/evaluation_results_16_04_2024/bmoe-piecewiselinear",
            "exp/mlp",
            "exp/mlp-piecewiselinear"
        ]

    output_path = "stat/"  # Replace with your desired output folder path

    dfs = []
    for base_path in base_paths:
        report_files = find_report_files(base_path)
        if not report_files:
            print("No report.json files found.")
            return

        # data_scores = extract_scores(report_files)
        if 'gumbel' in base_path:
            for n in n_bayesian_ensembles:
                if (type == 'advanced') and n != 10:
                    continue
                be_data_scores = extract_scores(report_files, n)
                dfs.append(calculate_statistics(be_data_scores))
        else:
            data_scores = extract_scores(report_files)

            if not data_scores:
                print("No scores extracted from the report.json files.")
                return

            dfs.append(calculate_statistics(data_scores))

    df = pd.concat(dfs)
    if type == 'x2':
        moe = 'MoE_x2'
        emoe = 'E+MoE_x2'
    elif type == 'gumbel':
        # moe = 'BMoE_gumbel'
        # emoe = 'E+BMoE_gumbel'
        moe = 'MoE'
        emoe = 'E+MoE'
    else:
        moe = 'MoE'
        emoe = 'E+MoE'

    if type not in ['x2', 'gumbel', 'advanced']:
        df = df.loc[df['model_type'].isin(['MoE', 'E+MoE', 'MLP', 'E+MLP'])]

    # calculated_datasets = df.loc[df['model_type'] == 'E+BMoE_adapter_sigmoid_tabm_mini']['dataset_name'].values
    # calculated_datasets = df.loc[df['model_type'] == 'E+BMoE_adapter_sigmoid']['dataset_name'].values
    calculated_datasets = df.loc[df['model_type'] == 'E+BMoE_gumbel_10']['dataset_name'].values
    # calculated_datasets = df.loc[df['model_type'] == 'E+BMoE_top_8_64_adapter_sigmoid_tabm_mini']['dataset_name'].values
    # calculated_datasets = df.loc[df['model_type'] == 'E+BMoE_big']['dataset_name'].values

    # calculated_datasets = df.loc[df['model_type'] == moe]['dataset_name'].values
    # calculated_datasets_emoe = df.loc[df['model_type'] == emoe]['dataset_name'].values
    # print('MoE - E+MoE')
    # print(np.setdiff1d(calculated_datasets, calculated_datasets_emoe))

    # print('E+MoE - MoE')
    # print(np.setdiff1d(calculated_datasets_emoe, calculated_datasets))
    all_datasets = df.loc[df['model_type'] == 'E+MLP']['dataset_name'].values
    print(f'calculated datasets:{len(calculated_datasets)}')
    print(calculated_datasets)
    print(f'missing datasets: {len(np.setdiff1d(all_datasets, calculated_datasets))}')
    print(np.setdiff1d(all_datasets, calculated_datasets))
    df['dataset_name'] = df['dataset_name'].str.replace(':data/', '', regex=True)
    df.loc[df['dataset_name'] == 'covtype', 'dataset_name'] = 'covtype2'
    gbdt_datasets = df.loc[df['model_type'] == 'catboost']['dataset_name'].values
    print(f'missing datasets gbdt: {np.setdiff1d(calculated_datasets, gbdt_datasets)}')

    df = df.loc[df['dataset_name'].isin(calculated_datasets)]
    df['dataset_name'] = df['dataset_name'].apply(lambda x: f"{x}_" if x.endswith('higgs-small') else x)
    df['dataset_index'] = df['dataset_name'].apply(lambda x: x.split('-')[-1])

    group_columns = ['dataset_index', 'model_type']
    # Get the remaining columns for aggregation
    agg_columns = df.columns.difference(group_columns)
    df = df.groupby(group_columns)[agg_columns].agg(
        lambda x: x.mean() if pd.api.types.is_numeric_dtype(x) else x.iloc[0]
    ).reset_index()  # .assign(
    #  rank=lambda x: x.groupby('dataset_index')['test_score_mean'].rank(ascending=False,
    #                                                                   method='min')).reset_index()  # ,
    #     rank_10=lambda x: x.groupby('dataset_index')['test_score_10_mean'].rank(ascending=False, method='min')).reset_index()
    # print(df.columns)
    # Add the backbone_type_winner column

    df = rank_models(df)

    winners = df[df['rank'] == 1].groupby('dataset_index').agg({'model_type': list})
    winners['n_winners'] = winners['model_type'].apply(lambda x: len(x))
    winners['model_type'] = winners['model_type'].apply(lambda x: x[0])
    winners.loc[winners['n_winners'] > 1, 'model_type'] = None
    winners = pd.Series(winners['model_type'], index=winners.index)
    df['model_type_winner'] = df['dataset_index'].map(winners)
    df['task'] = 'classification'
    df.loc[df['test_score_mean'] < 0, 'task'] = 'regression'
    # print(df.columns)
    # if type == 'x2':
    #     output_file = "model_scores_summary_17_01_2025_x2.xlsx"
    # elif type == 'gumbel':
    #     output_file = "model_scores_summary_17_01_2025_gumbel.xlsx"
    # else:
    #     output_file = "model_scores_summary_17_01_2025.xlsx"
    if type == 'advanced':
        output_file = 'model_scores_summary_advanced_05_03_2025.xlsx'
    else:
        assert False
    df['time_mean'] = df['time_mean'].astype(str)
    df['time_std'] = df['time_std'].astype(str)
    output_file_mini = output_file.split('.')[0] + '_mini.xlsx'
    output_file = save_to_excel(df, output_path, output_file)
    print(f"Summary saved to: {output_file}")
    print(df.shape)

    print(output_file_mini)
    save_to_excel(df[['dataset_name', 'model_type', 'model_type_winner', 'rank', 'test_score_mean', 'test_score_std',
                      'n_parameters_mean']], output_path, output_file_mini)
    # Calculate the average rank for each backbone_type

    avg_rank = df.groupby('model_type')['rank'].mean()

    print()

    # Print the result
    print('avg rank:')
    print(avg_rank)

    avg_rank_task = df.groupby(['model_type', 'task'])['rank'].mean()

    # Print the result
    print('avg rank task:')
    print(avg_rank_task)

    print(df.groupby('model_type').size())
    print('rank=1')
    print(df.loc[df['rank'] == 1].groupby('model_type').size())
    # print(df['model_type_winner'])
    print('winners:')
    print(winners.value_counts())
    # score_emoe = df.loc[df['model_type'] == 'E+MoE', 'test_score_mean']
    # score_emlp = df.loc[(df['model_type'] == 'E+MLP') & (df['dataset_name'].isin(calculated_datasets_emoe)) , 'test_score_mean']
    # print(score_emlp)
    # print(score_emoe.values / score_emlp.values)
    print('n_parameters mean:')
    print(df.groupby('model_type').agg({'n_parameters_mean': 'mean'}))
    print('n_parameters median:')
    print(df.groupby('model_type').agg({'n_parameters_mean': 'median'}))
    print(f"number of unique datasets: {len(df['dataset_index'].unique())}")

    print('rank sum:')
    print(df.groupby('model_type').agg({'rank': 'sum'}))

    print('rank mean:')
    print(df.groupby('model_type').agg({'rank': 'mean'}))

    print('rank std:')
    print(df.groupby('model_type').agg({'rank': 'std'}))

    print('length:')
    print(df.groupby('model_type').agg({'rank': len}))
    #
    # print(df.loc[df['model_type'] == 'E+BMoE_big', ['dataset_index', 'rank']].set_index('dataset_index') - df.loc[
    #     df['model_type'] == 'E+BMoE_adapter_sigmoid', ['dataset_index', 'rank']].set_index('dataset_index'))

    if type == 'gumbel':
        print('E+MoE - E+BMoE_gumbel')
        # print(df.loc[df['model_type'] == 'E+MoE', 'rank'].values - df.loc[
        #     df['model_type'] == 'E+BMoE_gumbel_10', 'rank'].values)
        print('E+MLP - E+BMoE_gumbel')
        print(df.loc[df['model_type'] == 'E+MLP', ['dataset_index', 'rank']].set_index('dataset_index') - df.loc[
            df['model_type'] == 'E+BMoE_gumbel_10', ['dataset_index', 'rank']].set_index('dataset_index'))


if __name__ == "__main__":
    main()
