import pandas as pd
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import re

SOURCE_CSV_PATH = "/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/train_cheXbert-myCopy1_with_valid_clean.csv"
DESTINATION_FOLDER = "./my_splits/"

# Define stratification columns
STRATIFY_COLS_FROM_GROUND = ['Sex', 'Age Group']
STRATIFY_COLS_FROM_TEST = ['Sex', 'Age Group']
STRATIFY_COLS_FROM_VALID = ['Sex', 'Age Group']

# percentage of rare subgroup of interest samples to include in the train, validation, and test sets
percentage_to_include = 0.3

def extract_patient_id(path):
    """Extracts the patient ID from the file path."""
    match = re.search(r'patient(\d+)', path)
    if match:
        return match.group(1)
    else:
        return None

def split_data(data, DESTINATION_FOLDER):
    age_ranges = [(0, 20), (20, 40), (80, float('inf'))]


    print("Splitting into ground truth...")
    ground_truth_data = pd.concat([data[(data['Age'] >= range_[0]) & (data['Age'] < range_[1])] for range_ in age_ranges]).copy()
    remaining_data = data[~data['Age'].isin(ground_truth_data['Age'])].copy()

 
    print("Assigning rare subgroup samples to the remaining splits...")
    rare_subgroup_samples_to_include, remaining_ground_truth_data = train_test_split(
        ground_truth_data,
        test_size=1 - percentage_to_include,
        random_state=42,
        stratify=ground_truth_data[STRATIFY_COLS_FROM_GROUND]
    )


    remaining_data = pd.concat([remaining_data, remaining_ground_truth_data])

  
    print("Splitting remaining data into train, validation, and test sets...")
    train_valid, test = train_test_split(
        remaining_data,
        test_size=0.15,
        random_state=42,
        stratify=remaining_data[STRATIFY_COLS_FROM_TEST]
    )

    train, valid = train_test_split(
        train_valid,
        test_size=0.18,
        random_state=42,
        stratify=train_valid[STRATIFY_COLS_FROM_VALID]
    )

    # esure no patient is shared across splits
    for patient in tqdm(data['Patient_ID'].unique()):
        splits = [filename.split('_')[0] for filename in data[data['Patient_ID'] == patient]['Path']]
        if len(set(splits)) > 1:
            chosen_split = random.choice(splits)
            data.loc[data['Patient_ID'] == patient, 'Path'] = data[data['Patient_ID'] == patient]['Path'].apply(lambda x: x if chosen_split in x else None)
    
    # dop rows with None paths
    data.dropna(subset=['Path'], inplace=True)

    train_csv_path = os.path.join(DESTINATION_FOLDER, "train.csv")
    valid_csv_path = os.path.join(DESTINATION_FOLDER, "valid.csv")
    test_csv_path = os.path.join(DESTINATION_FOLDER, "test.csv")
    ground_truth_csv_path = os.path.join(DESTINATION_FOLDER, "ground_truth.csv")

    print(f"Saving .csv files in {DESTINATION_FOLDER}")
    train.to_csv(train_csv_path, index=False)
    valid.to_csv(valid_csv_path, index=False)
    test.to_csv(test_csv_path, index=False)
    rare_subgroup_samples_to_include.to_csv(ground_truth_csv_path, index=False)

def do_it_main():

    data = pd.read_csv(SOURCE_CSV_PATH)

    data['Patient_ID'] = data['Path'].apply(extract_patient_id)

    # slit data
    split_data(data, DESTINATION_FOLDER)

if __name__ == "__main__":
    do_it_main()

