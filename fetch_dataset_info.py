import os
import json
import pandas as pd


def process_info_files(base_path="data"):
    """
    Processes info.json files in the given base path and extracts dataset details.

    Args:
        base_path (str): The base directory where datasets are located.

    Returns:
        list[dict]: A list of dictionaries containing parsed info.json data.
    """
    dataset_details = []

    # Iterate through each dataset folder
    for dataset_name in os.listdir(base_path):
        dataset_path = os.path.join(base_path, dataset_name)
        info_file_path = os.path.join(dataset_path, "info.json")

        # Check if the info.json file exists
        if os.path.isfile(info_file_path):
            try:
                # Read and parse the info.json file
                with open(info_file_path, "r") as f:
                    data = json.load(f)
                    dataset_details.append(data)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error reading {info_file_path}: {e}")

    return dataset_details


def save_to_excel(data, output_file="datasets_info.xlsx"):
    """
    Saves the list of dictionaries into an Excel file.

    Args:
        data (list[dict]): The data to be saved.
        output_file (str): The output Excel file path.
    """
    # Convert the list of dictionaries into a pandas DataFrame
    df = pd.DataFrame(data)

    # Save the DataFrame to an Excel file
    df.to_excel(output_file, index=False)
    print(f"Data saved to {output_file}")


if __name__ == "__main__":
    # Set the base path to your datasets
    base_path = "data"

    # Process all info.json files
    datasets_info = process_info_files(base_path)

    # Save the parsed data to an Excel file
    save_to_excel(datasets_info, output_file="datasets_info.xlsx")
