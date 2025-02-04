import pandas as pd
from sklearn.model_selection import train_test_split


#input_csv = './ground_truth_refit/refit_for_attribute_classifier_ground_truth.csv'  # Replace with the correct path
#input_csv = './test_refit/refit_for_attribute_classifier_test.csv'
input_csv = './test_refit_latest/refit_for_attribute_classifier_test.csv'
df = pd.read_csv(input_csv)


STRATIFY_COLS = ['Sex bin', 'Age Group']

# Step 1: Ensure that no patient is shared across splits by grouping by Patient ID
# We'll group by 'Patient ID', and then stratify using the 'Sex bin' and 'Age Group'

# get unique patients
unique_patients = df['Patient ID'].unique()


grouped_df = df.groupby('Patient ID').first().reset_index()  # One row per patient for stratification

# perform train/val/test split
train_val_patients, test_patients = train_test_split(
    grouped_df,
    test_size=0.10,  # First split 10% for test
    random_state=42,
    stratify=grouped_df[STRATIFY_COLS]
)

train_patients, val_patients = train_test_split(
    train_val_patients,
    test_size=0.11,  # 11% of remaining for val, results in approx 10% of total
    random_state=42,
    stratify=train_val_patients[STRATIFY_COLS]
)

# filter the original dataframe for each split
train_df = df[df['Patient ID'].isin(train_patients['Patient ID'])]
val_df = df[df['Patient ID'].isin(val_patients['Patient ID'])]
test_df = df[df['Patient ID'].isin(test_patients['Patient ID'])]

# save the resulting splits to CSV files
#train_df.to_csv('./ground_truth_splits/gt_attr_train.csv', index=False)  # Replace with the desired output path
#val_df.to_csv('./ground_truth_splits/gt_attr_val.csv', index=False)
#test_df.to_csv('./ground_truth_splits/gt_attr_test.csv', index=False)

train_df.to_csv('./test_splits_latest/test_attr_train.csv')
val_df.to_csv('./test_splits_latest/test_attr_val.csv')
test_df.to_csv('./test_splits_latest/test_attr_test.csv')



print("Splitting complete. CSVs saved.")

