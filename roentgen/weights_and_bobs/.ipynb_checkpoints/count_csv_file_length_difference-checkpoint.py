import os
import pandas as pd

# directories holding csv files
dir_1 = "/workspace/my_auxiliary_persistent/upsampled_test_set/csv_files/test_combined_with_upsamples_upsamples/"
dir_2 = "/workspace/my_auxiliary_persistent/upsampled_test_set/csv_files/upsamples_to_add_to_test/"

# list of all CSV files in both directories
csv_files_dir_1 = [f for f in os.listdir(dir_1) if f.endswith('.csv')]
csv_files_dir_2 = [f for f in os.listdir(dir_2) if f.endswith('.csv')]

def print_csv_info(directory, filenames):
    csv_info = {}
    for filename in filenames:
        file_path = os.path.join(directory, filename)
        df = pd.read_csv(file_path)
        row_count = len(df)
        csv_info[filename] = row_count
        print(f"Filename: {filename}, Row count: {row_count}")
    return csv_info

print("CSV files in Directory 1:")
csv_info_dir_1 = print_csv_info(dir_1, csv_files_dir_1)

print("\nCSV files in Directory 2:")
csv_info_dir_2 = print_csv_info(dir_2, csv_files_dir_2)

# pair files based on their name prefixes
def pair_files_by_prefix(csv_info_1, csv_info_2, prefix_length=10):
    paired_files = []
    for file_1 in csv_info_1:
        for file_2 in csv_info_2:
            if file_1[:prefix_length] == file_2[:prefix_length]:
                paired_files.append((file_1, csv_info_1[file_1], file_2, csv_info_2[file_2]))
    return paired_files

# pair files by their prefixes (adjust the prefix_length as needed)
paired_files = pair_files_by_prefix(csv_info_dir_1, csv_info_dir_2, prefix_length=10)


print("\nPaired CSV files:")
for pair in paired_files:
    print(f"Dir 1: {pair[0]} (Rows: {pair[1]}), Dir 2: {pair[2]} (Rows: {pair[3]})")

