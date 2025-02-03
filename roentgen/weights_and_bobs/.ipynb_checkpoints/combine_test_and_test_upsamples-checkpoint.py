import pandas as pd
import os

# Directories for the original and upsampled CSV files
# original 3
#upsamples_dir = '/workspace/my_auxiliary_persistent/upsampled_test_set/csv_files/upsamples_to_add_to_test/'
#original_dir = '/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/'
#output_dir = '/workspace/my_auxiliary_persistent/upsampled_test_set/csv_files/test_combined_with_upsamples/'

upsamples_dir = '/workspace/my_auxiliary_persistent/upsampled_test_set/csv_files/upsamples_to_add_to_upsamples/'
original_dir = '/workspace/my_auxiliary_persistent/upsampled_test_set/csv_files/upsamples_to_add_to_test/'
output_dir = '/workspace/my_auxiliary_persistent/upsampled_test_set/csv_files/synth_test_set/'

#out dir if it aint there
os.makedirs(output_dir, exist_ok=True)

# list of additional upsampled CSV files
upsampled_files = [f for f in os.listdir(upsamples_dir) if f.endswith('.csv')]

# iterate through  upsampled files and combine them with the originals
for upsampled_file in upsampled_files:
    # construct the corresponding original filename
    # for the original directory trio
    #original_file = upsampled_file.replace('_to_be_added_upsamples.csv', '.csv')
    # for the latter directory trio
    original_file = upsampled_file.replace('_to_be_added_to_full_synth_upsamples.csv', '.csv')
    
    # define the full paths for original and upsampled files
    original_filepath = os.path.join(original_dir, original_file)
    upsampled_filepath = os.path.join(upsamples_dir, upsampled_file)
    
    # check if the original file exists
    if os.path.exists(original_filepath):
        # read both CSV files
        original_df = pd.read_csv(original_filepath)
        upsampled_df = pd.read_csv(upsampled_filepath)
        
        # concatenate the original and upsampled dfs
        combined_df = pd.concat([original_df, upsampled_df], ignore_index=True)
        
        # save the combined DataFrame to the output directory
        combined_output_filepath = os.path.join(output_dir, original_file)
        combined_df.to_csv(combined_output_filepath, index=False)
        
        print(f"Combined {original_file} with {upsampled_file} and saved to {combined_output_filepath}.")
    else:
        print(f"Original file {original_file} does not exist. Skipping.")

print("All applicable files have been combined successfully.")

