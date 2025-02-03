import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


#CSV_FILE = "/workspace/persistent/code/roentgen/weights_and_bobs/valid_csv_from_my_splits/valid.csv" # OLD
CSV_FILE = "/workspace/my_auxiliary_persistent/retrain_roentgen/csvs/source_test_csv/test.csv" # NEW RETRAIN ON TEST
PATH_TO_IMAGES = "path/to/images/" # currently doesnt do anything


class CustomChestXRayDataset:
    def __init__(self, csv_file, img_dir):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir

    def split_train_valid(self, test_size=0.1, random_state=42):
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        for train_idx, valid_idx in splitter.split(self.data, self.data[['Sex', 'Age Group']]):
            train_data = self.data.iloc[train_idx]
            valid_data = self.data.iloc[valid_idx]
        return train_data, valid_data


dataset = CustomChestXRayDataset(CSV_FILE, PATH_TO_IMAGES)


train_data, valid_data = dataset.split_train_valid()

# Save train and validation CSV files
# OLD
#train_data.to_csv("/workspace/persistent/code/roentgen/weights_and_bobs/diffusion_splits/train_from_valid.csv", index=False)
#valid_data.to_csv("/workspace/persistent/code/roentgen/weights_and_bobs/diffusion_splits/valid_from_valid.csv", index=False)

# NEW RETRAIN ROENTGEN ON TEST
train_data.to_csv("/workspace/my_auxiliary_persistent/retrain_roentgen/csvs/diffusion_splits/train_from_test.csv", index=False)
valid_data.to_csv("/workspace/my_auxiliary_persistent/retrain_roentgen/csvs/diffusion_splits/valid_from_test.csv", index=False)
