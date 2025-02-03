import os
import shutil
import pandas as pd
from tqdm import tqdm

# define the source prefix and destination folder
source_prefix = "/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/"
#destination_folder = "/workspace/my_auxiliary_persistent/upsampled_test_set/imgs/"
destination_folder = "/workspace/my_auxiliary_persistent/retrain_roentgen/generated_test_sets/imgs/" # retrain roentgen on test set 

#csv_file = "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/test.csv"
csv_file = "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/valid.csv"


df = pd.read_csv(csv_file)

# Fcopy images while preserving the directory structure
def copy_images():
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Copying images", unit="image"):

        relative_path = row['Path']

        source_path = os.path.join(source_prefix, relative_path)

        destination_path = os.path.join(destination_folder, relative_path)

        # create any necessary directories in the destination path
        destination_dir = os.path.dirname(destination_path)
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)

        # copy the image from source to destination
        if os.path.exists(source_path):  # Check if the source file exists
            shutil.copy2(source_path, destination_path)  
        else:
            print(f"Source image not found: {source_path}")

if __name__ == "__main__":
    copy_images()
    print("Image copying completed.")

