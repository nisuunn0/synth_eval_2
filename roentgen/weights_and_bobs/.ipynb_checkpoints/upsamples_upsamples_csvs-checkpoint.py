# BASED ON upsample_test_csv.py.
# Same operating principle, but new source and destination csv directories, new logic for determining num samples to upsample by

import os
import pandas as pd
import random
from collections import Counter
from tqdm import tqdm


csv_directory = "/workspace/my_auxiliary_persistent/upsampled_test_set/csv_files/upsamples_to_add_to_test/"
destination_directory = "/workspace/my_auxiliary_persistent/upsampled_test_set/csv_files/upsamples_to_add_to_upsamples/"
old_csv_directory = "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/"


def generate_synthetic_path(patient_id_counter):
    return f"further_upsampled_synth_portion/patient{str(patient_id_counter).zfill(5)}/study1/img0.jpg"


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

    # randomly assign an age between the range given by the "Age Group"
    age_group = new_row['Age Group']
    if age_group == '20-40':
        new_row['Age'] = random.randint(20, 40)
    elif age_group == '0-20':
        new_row['Age'] = random.randint(0, 20)
    elif age_group == '80+':
        new_row['Age'] = random.randint(80, 99)

    # apply "No Finding" constraints
    new_row = apply_no_finding_constraints(new_row)

    # Handle "AP/PA" and "Frontal/Lateral" relationship
    if new_row['Frontal/Lateral'] == 'Lateral':
        new_row['AP/PA'] = '0'
    else:
        new_row['AP/PA'] = random.choice(['AP', 'PA'])

    return new_row


def load_csv_files(prefix, directory):
    return {f: pd.read_csv(os.path.join(directory, f)) for f in os.listdir(directory) if f.startswith(prefix)}


test_files = load_csv_files('test_', csv_directory)
# Load the corresponding test files from the old directory
old_test_files = load_csv_files('test_', old_csv_directory)


disease_columns = [
    'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion',
    'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
    'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
]


patient_id_counter = 850000


for test_filename, test_df in tqdm(test_files.items(), desc="Upsampling Test Files"):
 
    #old_test_filename = test_filename.replace('test_', 'test_')
    #print("old_test_filename: "  + str(old_test_filename))
    
    old_test_filename = test_filename.replace('_to_be_added_upsamples', '')
    print(f"old_test_filename: {old_test_filename}")

   
    if old_test_filename in old_test_files:
        print("old exists")
        old_test_df = old_test_files[old_test_filename]
        
       
        rows_to_add = len(old_test_df)
        print(f"Upsampling {test_filename}: Adding {rows_to_add} rows.")
        
        test_df['stratify_col'] = test_df[disease_columns + ['Sex', 'Age Group']].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
        distribution = test_df['stratify_col'].value_counts(normalize=True)

        new_rows = []
       
        for _ in tqdm(range(rows_to_add), desc=f"Generating new rows for {test_filename}"):
            stratified_sample = distribution.sample(n=1, weights=distribution).index[0]
            
            sampled_row = test_df[test_df['stratify_col'] == stratified_sample].sample(n=1).iloc[0]

            # generate a new synthetic row
            new_sample = generate_new_sample(sampled_row, patient_id_counter)
            new_rows.append(new_sample)

            # increment the patient ID counter for unique paths
            patient_id_counter += 1

 
        new_rows_df = pd.DataFrame(new_rows)

   
        output_filename = os.path.join(destination_directory, test_filename.replace('.csv', '_to_be_added_to_full_synth_upsamples.csv'))
        new_rows_df.to_csv(output_filename, index=False)
        print(f"Saved {rows_to_add} new rows to {output_filename}.")

print("Upsampling complete for all test files.")

