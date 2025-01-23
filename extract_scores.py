import os
import json
import pandas as pd

bayesian_ensemble_file_name = 'report_bayes_ensemble.json'
n_bayesian_ensembles = [1, 5, 10, 100]
import numpy as np

type = 'sssx2'
def find_report_files(base_path):
    """
    Find all `report.json` files matching the pattern */0-evaluation/{n_seed}/report.json.
    """
    report_files = []
    for root, _, files in os.walk(base_path):
        for file in files:
            if file == "report.json" and "0-evaluation" in root:
                report_files.append(os.path.join(root, file))
    return report_files


def extract_scores(report_files):
    """
    Extract scores from `report.json` files and group them by dataset_name.
    """
    data_scores = {}
    for file_path in report_files:

        with open(file_path, 'r') as f:
            report = json.load(f)
            dataset_path = report['config']['data']['path'].removeprefix('data/')
            scores = {
                "train_score": report['metrics']['train']['score'],
                "val_score": report['metrics']['val']['score'],
                "test_score": report['metrics']['test']['score'],
                "n_parameters": report['n_parameters'],
            }
            for n in n_bayesian_ensembles:
                scores['test_score_' + str(n)] = report['metrics']['test']['score']
            dir_path = os.path.dirname(file_path)
            be_file_name = os.path.join(dir_path, bayesian_ensemble_file_name)
            if os.path.exists(be_file_name):
                with open(be_file_name, 'r') as f:
                    report_be = json.load(f)
                for n in n_bayesian_ensembles:
                    scores['test_score_' + str(n)] = report_be['metrics'][str(n)]['test']['score']
            model_type = report['config']['model']['backbone']['type']
            if (model_type == 'BMoE') and report['config']['model']['backbone']['gating_type'] == 'standard':
                model_type = 'MoE'
            if 'evaluation_results_x2' in file_path:
                model_type += '_x2'
            if 'config' in report and 'model' in report['config'] and \
                    'num_embeddings' in report['config']['model']:
                model_type = f"E+{model_type}"

            scores['model_type'] = model_type
            if dataset_path not in data_scores:
                data_scores[dataset_path] = {"train_score": [], "val_score": [], "test_score": [],
                                             'n_parameters': [], 'model_type': []}
                for n in n_bayesian_ensembles:
                    data_scores[dataset_path]['test_score_' + str(n)] = []
            for key in scores:
                data_scores[dataset_path][key].append(scores[key])

    return data_scores


def calculate_statistics(data_scores):
    """
    Calculate mean and standard deviation of scores for each dataset_name.
    """
    records = []
    for dataset_path, scores in data_scores.items():
        record = {"dataset_name": dataset_path, 'model_type': scores['model_type'][0]}
        be_test = ['test_score_' + str(n) for n in n_bayesian_ensembles]
        for metric in ["train_score", "val_score", "test_score", 'n_parameters'] + be_test:
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


import pandas as pd


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
    base_paths = [

        "exp/results/evaluation_results_16_04_2024/moe",
        "exp/results/evaluation_results_x2/moe",
        # "exp/results/evaluation_results_16_04_2024/bmoe",
        "exp/results/evaluation_results_16_04_2024/moe-piecewiselinear",
        "exp/results/evaluation_results_x2/moe-piecewiselinear",
        # "exp/results/evaluation_results_16_04_2024/bmoe-piecewiselinear",
        "exp/mlp",
        "exp/mlp-piecewiselinear"
    ]  # Replace with your dataset folder path
    output_path = "stat/"  # Replace with your desired output folder path

    dfs = []
    for base_path in base_paths:
        report_files = find_report_files(base_path)
        if not report_files:
            print("No report.json files found.")
            return

        data_scores = extract_scores(report_files)
        if not data_scores:
            print("No scores extracted from the report.json files.")
            return

        dfs.append(calculate_statistics(data_scores))

    df = pd.concat(dfs)
    if type == 'x2':
        moe = 'MoE_x2'
        emoe = 'E+MoE_x2'
    else:
        moe = 'MoE'
        emoe = 'E+MoE'
    if type != 'x2':
        df = df.loc[df['model_type'].isin(['MoE', 'E+MoE', 'MLP', 'E+MLP'])]
    calculated_datasets = df.loc[df['model_type'] == moe]['dataset_name'].values
    calculated_datasets_emoe = df.loc[df['model_type'] == emoe]['dataset_name'].values
    print('MoE - E+MoE')
    print(np.setdiff1d(calculated_datasets, calculated_datasets_emoe))

    print('E+MoE - MoE')
    print(np.setdiff1d(calculated_datasets_emoe, calculated_datasets))
    all_datasets = df.loc[df['model_type'] == 'MLP']['dataset_name'].values
    print('missing datasets:')
    print(np.setdiff1d(all_datasets, calculated_datasets))

    df = df.loc[df['dataset_name'].isin(calculated_datasets)]
    df['dataset_index'] = df['dataset_name'].apply(lambda x: x.split('-')[-1])

    group_columns = ['dataset_index', 'model_type']
    # Get the remaining columns for aggregation
    agg_columns = df.columns.difference(group_columns)
    df = df.groupby(group_columns)[agg_columns].agg(
        lambda x: x.mean() if pd.api.types.is_numeric_dtype(x) else x.iloc[0]
    ).assign(
        rank=lambda x: x.groupby('dataset_index')['test_score_mean'].rank(ascending=False,
                                                                          method='min')).reset_index()  # ,
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
    output_file = save_to_excel(df, output_path, "model_scores_summary_17_01_2025_x2.xlsx")
    print(f"Summary saved to: {output_file}")
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
    print(df.loc[df['rank'] == 1].groupby('model_type').size())
    # print(df['model_type_winner'])
    print(winners.value_counts())
    # score_emoe = df.loc[df['model_type'] == 'E+MoE', 'test_score_mean']
    # score_emlp = df.loc[(df['model_type'] == 'E+MLP') & (df['dataset_name'].isin(calculated_datasets_emoe)) , 'test_score_mean']
    # print(score_emlp)
    # print(score_emoe.values / score_emlp.values)
    print('n_parameters:')
    print(df.groupby('model_type').agg({'n_parameters_mean': 'mean'}))

if __name__ == "__main__":
    main()
