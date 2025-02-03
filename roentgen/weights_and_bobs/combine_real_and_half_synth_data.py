import pandas as pd
import os


REAL_IMAGE_PATH_PREFIX = "/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/"
SYNTHETIC_IMAGE_PATH_PREFIX = "/workspace/my_auxiliary_persistent/retrain_roentgen/full_test_set_for_metric_assessment/checkpoint-44000/"

# Columns related to diseases
disease_columns = [
    'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion',
    'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
    'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
]

# load the real dataset csv files
def load_real_csv_files(csv_path):
    return pd.read_csv(csv_path)

# stratified sampling to select rows from real dataset
def stratified_sample(df, num_samples):
    # create a stratification column to group based on disease columns and other attributes
    df['stratify_col'] = df[disease_columns + ['Sex', 'Age Group']].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
    
    # calculate normalized distribution of  stratify_col
    distribution = df['stratify_col'].value_counts(normalize=True)
    
    # sample the required number of rows based on the stratified distribution
    sampled_rows = []
    for _ in range(num_samples):
        stratified_sample = distribution.sample(n=1, weights=distribution).index[0]
        sampled_row = df[df['stratify_col'] == stratified_sample].sample(n=1).iloc[0]
        sampled_rows.append(sampled_row)
    
    return pd.DataFrame(sampled_rows)

# main function to perform the sampling and merge with the real data
def generate_combined_dataset(real_df, synthetic_ratio=1/2):
    # update paths in the real dataset to include the real image path prefix
    real_df['Path'] = real_df['Path'].apply(lambda p: os.path.join(REAL_IMAGE_PATH_PREFIX, p))
    
    #calc how many rows to sample
    num_real_samples = len(real_df)
    num_synthetic_samples = int(num_real_samples * synthetic_ratio)
    
    # stratified sampling to get synthetic data
    synthetic_df = stratified_sample(real_df, num_synthetic_samples)
    
    # update paths in the synthetic dataset to include the synthetic image path prefix
    synthetic_df['Path'] = synthetic_df['Path'].apply(lambda p: os.path.join(SYNTHETIC_IMAGE_PATH_PREFIX, p))
    
    # drop stratify_col to clean up
    synthetic_df.drop(columns=['stratify_col'], inplace=True, errors='ignore')

    # combine real and synthetic datasets
    combined_df = pd.concat([real_df, synthetic_df], ignore_index=True)
    
    return combined_df

# load the real dataset (replace with actual CSV file path)
real_csv_path = "/workspace/my_auxiliary_persistent/retrain_roentgen/train_on_test/test_split_to_train_and_valid/train_split.csv"
real_df = load_real_csv_files(real_csv_path)

# generate combined dataset with desired synthetic-to-real ratio (e.g., 1:2)
combined_df = generate_combined_dataset(real_df, synthetic_ratio=1/4) #3/4)#1/2)


combined_df.to_csv('/workspace/my_auxiliary_persistent/retrain_roentgen/train_on_test/real_and_synth_data_together_csvs/full_real_plus_one_quarter_synth.csv', index=False)

print(f"Combined dataset saved with {len(combined_df)} rows.")

