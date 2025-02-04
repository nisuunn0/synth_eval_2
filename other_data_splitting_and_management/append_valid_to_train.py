import pandas as pd
from tqdm import tqdm
import os


RESULT_CSV_PATH = "/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/train_cheXbert-myCopy1_with_valid.csv"
VALID_PATH = "/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0_batch_1_validate_&_csv/valid.csv"


def correct_path(path):

    return "CheXpert-v1.0_batch_1_validate_&_csv" + path[len("CheXpert-v1.0"):]

def add_valid_to_all_train_csv(result_csv_path, valid_path):
    # read csv
    result_df = pd.read_csv(result_csv_path)

    # read val csv
    valid_df = pd.read_csv(valid_path)

    # correct path in val df
    valid_df['Path'] = valid_df['Path'].apply(correct_path)

    # append valid data to existing data
    result_df = pd.concat([result_df, valid_df], ignore_index=True)

    # write back to csv
    result_df.to_csv(result_csv_path, index=False)

def do_it_all():
    print("Current directory: " + os.getcwd())
    add_valid_to_all_train_csv(RESULT_CSV_PATH, VALID_PATH)

if __name__ == "__main__":
    do_it_all()

