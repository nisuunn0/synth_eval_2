# SAME AS split_valid.py BUT FOR TRAINING DOWNSTREAM CLASSIFIER ON EITHER REAL OR SYNTHETIC (OR COMBO) TEST SET.

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


TEST_DF_PATH = "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/test.csv"
DEST_FOLDER = "/workspace/my_auxiliary_persistent/retrain_roentgen/train_on_test/test_split_to_train_and_valid/"


class CustomChestXRayDataset:
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def split_train_valid(self, test_size=0.2, random_state=42):
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        for train_idx, valid_idx in splitter.split(self.data, self.data[['Sex', 'Age Group']]):
            train_data = self.data.iloc[train_idx]
            valid_data = self.data.iloc[valid_idx]
        return train_data, valid_data

# init datawesr
dataset = CustomChestXRayDataset(TEST_DF_PATH)

# ttrain val split
train_data, valid_data = dataset.split_train_valid(test_size=0.2)

# savee em yo
train_data.to_csv(f"{DEST_FOLDER}train_split.csv", index=False)
valid_data.to_csv(f"{DEST_FOLDER}valid_split.csv", index=False)

print("Train and validation splits have been saved successfully.")
