import pandas as pd
import os
from sklearn.model_selection import train_test_split

# old source_csv_path before cleaning
#SOURCE_CSV_PATH = "/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/train_cheXbert-myCopy1_with_valid.csv"
# new source_csv_path after cleaning (replace NaN and other non-positive cases with 0.0)
SOURCE_CSV_PATH = "/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/train_cheXbert-myCopy1_with_valid_clean.csv"
DESTINATION_FOLDER = "./my_splits/"

STRATIFY_COLS_FROM_GROUND= ['Sex', 'Age Group',] #'Atelectasis', 'Cardiomegaly', 'Edema'] #'Consolidation',] #'Edema', 'Pleural Effusion']
STRATIFY_COLS_FROM_TEST = ['Sex', 'Age Group',] #'Atelectasis', 'Cardiomegaly', 'Edema'] #'Consolidation',] # 'Edema', 'Pleural Effusion']
STRATIFY_COLS_FROM_VALID = ['Sex', 'Age Group',] #'Atelectasis', 'Cardiomegaly', 'Edema']  #'Consolidation',] # 'Edema', 'Pleural Effusion']

# percentage of rare subgroup of interest samples to include in the train, validation, and test sets
percentage_to_include = 0.3 #0.05  # Adjust as needed

def split_data(data, DESTINATION_FOLDER):
    age_ranges = [(0, 20), (20, 40), (80, float('inf'))]

    # split data into ground truth set and rest based on age ranges
    print("splitting into grouond truth")
    ground_truth_data = pd.concat([data[(data['Age'] >= range_[0]) & (data['Age'] < range_[1])] for range_ in age_ranges]).copy()
    remaining_data = data[~data['Age'].isin(ground_truth_data['Age'])].copy()
    
    
    # sample a portion of the rare subgroup of interest for the train, validation, and test sets
    print("giving back some rare subroup samples to the 3 remaining splits (train, valid, test)")
    print(ground_truth_data.head)
    rare_subgroup_samples_to_include, remaining_ground_truth_data = train_test_split(
        ground_truth_data, 
        test_size=1 - percentage_to_include, 
        random_state=42, 
        stratify=ground_truth_data[STRATIFY_COLS_FROM_GROUND]
    )

    # combine remaining rare subgroup samples with remaining data
    remaining_data = pd.concat([remaining_data, remaining_ground_truth_data])
    
    # split remaining data into train, validation, and test sets
    print("splitting remaining data into train, val, test")
    train_valid, test = train_test_split(
        remaining_data, 
        test_size=0.15, # 0.2, 
        random_state=42, 
        stratify=remaining_data[STRATIFY_COLS_FROM_TEST]
    )
    
    train, valid = train_test_split(
        train_valid, 
        test_size=0.18, #0.2, 
        random_state=42, 
        stratify=train_valid[STRATIFY_COLS_FROM_VALID]
    )
    
    # paths for the output csv files
    train_csv_path = os.path.join(DESTINATION_FOLDER, "train.csv")
    valid_csv_path = os.path.join(DESTINATION_FOLDER, "valid.csv")
    test_csv_path = os.path.join(DESTINATION_FOLDER, "test.csv")
    ground_truth_csv_path = os.path.join(DESTINATION_FOLDER, "ground_truth.csv")
    
    # save csv
    print(f"saving .csv files in {DESTINATION_FOLDER}")
    train.to_csv(train_csv_path, index=False)
    valid.to_csv(valid_csv_path, index=False)
    test.to_csv(test_csv_path, index=False)
    rare_subgroup_samples_to_include.to_csv(ground_truth_csv_path, index=False)

def do_it_main():
  
    data = pd.read_csv(SOURCE_CSV_PATH)

  
    split_data(data, DESTINATION_FOLDER)

if __name__ == "__main__":
    do_it_main()

