import os
import pandas as pd


csv_directory = "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/"

# ground truth does not have these, ignore them.
exclude_suffixes = ["40-80.csv", "40-80_Female.csv", "40-80_Male.csv"]

# get a list of files with a specific prefix, excluding certain suffixes
def get_csv_files_with_prefix(prefix):
    return [f for f in os.listdir(csv_directory) 
            if f.startswith(prefix) and f.endswith('.csv') and all(excl not in f for excl in exclude_suffixes)]

# get the list of ground truth and test CSV files
ground_truth_files = get_csv_files_with_prefix('ground_truth_')
test_files = get_csv_files_with_prefix('test_')

# sort the files to ensure matching order
ground_truth_files.sort()
test_files.sort()

def count_rows_in_csv(file_path):
    df = pd.read_csv(file_path)
    return len(df)

# loop over the files and print the row counts for matching pairs
for gt_file, test_file in zip(ground_truth_files, test_files):
    # esure the files have matching suffixes
    if gt_file.split('ground_truth_')[1] == test_file.split('test_')[1]:
        gt_file_path = os.path.join(csv_directory, gt_file)
        test_file_path = os.path.join(csv_directory, test_file)
        
        gt_row_count = count_rows_in_csv(gt_file_path)
        test_row_count = count_rows_in_csv(test_file_path)
        
     
        print(f"{gt_file}: {gt_row_count} samples, {test_file}: {test_row_count} samples")
    else:
        print(f"File mismatch: {gt_file} and {test_file} do not match.")

