import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns


#original_dir = '/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/'  # Directory with the original test CSV files
#upsampled_dir = '/workspace/my_auxiliary_persistent/upsampled_test_set/csv_files/upsamples_to_add_to_test/'  # Directory with the upsampled CSV files
#save_dir = '/workspace/my_auxiliary_persistent/upsampled_test_set/csv_files/plot_upsample_stats/'  # Directory to save comparison plots


original_dir = '/workspace/my_auxiliary_persistent/upsampled_test_set/csv_files/upsamples_to_add_to_test/'  # Directory with the original test CSV files
upsampled_dir = '/workspace/my_auxiliary_persistent/upsampled_test_set/csv_files/upsamples_to_add_to_upsamples/'  # Directory with the upsampled CSV files
save_dir = '/workspace/my_auxiliary_persistent/upsampled_test_set/csv_files/plot_upsample_stats_part_2/'  # Directory to save comparison plots



os.makedirs(save_dir, exist_ok=True)


disease_columns = [
    'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion',
    'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
    'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
]
attribute_columns = ['Sex', 'Age Group', 'Frontal/Lateral', 'AP/PA', 'No Finding']


def plot_distribution_comparison(original_df, upsampled_df, column, save_path):
    combined_df = pd.concat([original_df, upsampled_df])
    
    # create a 'source' column to differentiate between original and upsampled data
    combined_df['source'] = ['Original'] * len(original_df) + ['Upsampled'] * len(upsampled_df)
    

    plt.figure(figsize=(10, 6))
    sns.countplot(data=combined_df, x=column, hue='source', palette='Set2')
    plt.title(f'Distribution Comparison: {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    

    plt.savefig(save_path)
    plt.close()


def sanitize_filename(column_name):
    return column_name.replace('/', '_').replace('\\', '_')



for original_filename in os.listdir(original_dir):
    # filter files: only process files that start with "test_" and do not contain "40-80"
    if original_filename.startswith('test_') and '40-80' not in original_filename and original_filename.endswith('.csv'):
        # load the original and upsampled csvfiles
        original_filepath = os.path.join(original_dir, original_filename)
        original_df = pd.read_csv(original_filepath)
        
        # find corresponding upsampled file
        upsampled_filename = original_filename.replace('.csv', '_to_be_added_to_full_synth_upsamples.csv')
        upsampled_filepath = os.path.join(upsampled_dir, upsampled_filename)
        
        if os.path.exists(upsampled_filepath):
            upsampled_df = pd.read_csv(upsampled_filepath)
            
            # Compare each disease and attribute column
            for column in disease_columns + attribute_columns:
                #save_path = os.path.join(save_dir, f'{original_filename}_{column}_comparison.png')
                sanitized_column = sanitize_filename(column)  # Sanitize the column name for filenames
                save_path = os.path.join(save_dir, f'{original_filename}_{sanitized_column}_comparison.png')
                
                plot_distribution_comparison(original_df, upsampled_df, column, save_path)

print("Comparison plots saved successfully.")

