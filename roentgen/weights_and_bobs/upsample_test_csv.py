import os
import pandas as pd
import random
from collections import Counter
from tqdm import tqdm


csv_directory = "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/"
destination_directory = "/workspace/my_auxiliary_persistent/upsampled_test_set/csv_files/upsamples_to_add_to_test/"


def generate_synthetic_path(patient_id_counter):
    return f"upsampled_synth_portion/patient{str(patient_id_counter).zfill(5)}/study1/img0.jpg"


def apply_no_finding_constraints(row):
    if row['No Finding'] == 1:
        for col in disease_columns:
            row[col] = 0
    else:
        if all(row[col] == 0 for col in disease_columns):  # Ensure at least one disease is present if 'No Finding' == 0
            row[random.choice(disease_columns)] = 1
    return row


def generate_new_sample(existing_row, patient_id_counter):
    new_row = existing_row.copy()

    
    new_row['Path'] = generate_synthetic_path(patient_id_counter)

    
    age_group = new_row['Age Group']
    if age_group == '20-40':
        new_row['Age'] = random.randint(20, 40)
    elif age_group == '0-20':
        new_row['Age'] = random.randint(0, 20)
    
    elif age_group == '80+':
        new_row['Age'] = random.randint(80, 99)

    # Apply "No Finding" constraints
    new_row = apply_no_finding_constraints(new_row)

    # Handle "AP/PA" and "Frontal/Lateral" relationship
    if new_row['Frontal/Lateral'] == 'Lateral':
        new_row['AP/PA'] = '0'
    else:
        new_row['AP/PA'] = random.choice(['AP', 'PA'])

    return new_row


def load_csv_files(prefix):
    return {f: pd.read_csv(os.path.join(csv_directory, f)) for f in os.listdir(csv_directory) if f.startswith(prefix)}

ground_truth_files = load_csv_files('ground_truth_')
test_files = load_csv_files('test_')

# Columns related to diseases
disease_columns = [
    'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion',
    'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
    'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
]


patient_id_counter = 100000  # Initial synthetic patient ID, pretty arbitrary


for test_filename, test_df in tqdm(test_files.items(), desc="Upsampling Test Files"):
    # corresponding ground truth file (if no corresponding ground truth file exists, then the current iteration will be skipped)
    gt_filename = test_filename.replace('test_', 'ground_truth_')
    if gt_filename in ground_truth_files:
        gt_df = ground_truth_files[gt_filename]
        
        # calculate how many rows we need to add
        rows_to_add = len(gt_df) - len(test_df)
        if rows_to_add > 0:
            print(f"Upsampling {test_filename}: Adding {rows_to_add} rows.")

            # Stratified sampling based on disease columns
            # create a helper column in the test dataframe for stratified sampling based on unique combinations
            test_df['stratify_col'] = test_df[disease_columns + ['Sex', 'Age Group']].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
            
            # calculate the normalized distribution of the stratify_col
            distribution = test_df['stratify_col'].value_counts(normalize=True)

            new_rows = []
        
            for _ in tqdm(range(rows_to_add), desc=f"Generating new rows for {test_filename}"):
                # Sample a row using the stratified distribution
                stratified_sample = distribution.sample(n=1, weights=distribution).index[0]
                
                # Select a random row from the test dataframe that matches the stratified sample
                sampled_row = test_df[test_df['stratify_col'] == stratified_sample].sample(n=1).iloc[0]

                # generate a new synthetic row
                new_sample = generate_new_sample(sampled_row, patient_id_counter)
                new_rows.append(new_sample)

                # increment the patient ID counter for unique paths
                patient_id_counter += 1

            # create a DataFrame for new rows
            new_rows_df = pd.DataFrame(new_rows)

           
            output_filename = os.path.join(destination_directory, test_filename.replace('.csv', '_to_be_added_upsamples.csv'))
            new_rows_df.to_csv(output_filename, index=False)
            print(f"Saved {rows_to_add} new rows to {output_filename}.")
