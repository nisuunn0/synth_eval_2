import os
import pandas as pd
import random
from tqdm import tqdm


input_csv_path = "/workspace/my_auxiliary_persistent/retrain_roentgen/train_on_test/test_split_to_train_and_valid/train_split.csv"
output_directory = "/workspace/my_auxiliary_persistent/retrain_roentgen/train_on_test/additional_images/"

# generate synthetic path for new images
def generate_synthetic_path(patient_id_counter):
    return f"upsampled_synth_portion/patient{str(patient_id_counter).zfill(5)}/study1/img0.jpg"

# apply the constraint: "No Finding" column value rules
def apply_no_finding_constraints(row):
    if row['No Finding'] == 1:
        for col in disease_columns:
            row[col] = 0
    else:
        if all(row[col] == 0 for col in disease_columns):  # Ensure at least one disease is present if 'No Finding' == 0
            row[random.choice(disease_columns)] = 1
    return row

# generate a new upsampled row
def generate_new_sample(existing_row, patient_id_counter):
    new_row = existing_row.copy()

    new_row['Path'] = generate_synthetic_path(patient_id_counter)

    # randomly assign an age based on the range defined by the "Age Group"
    age_group = new_row['Age Group']
    if age_group == '20-40':
        new_row['Age'] = random.randint(20, 40)
    elif age_group == '0-20':
        new_row['Age'] = random.randint(0, 20)
    elif age_group == '40-80':  # Added missing range
        new_row['Age'] = random.randint(40, 80)
    elif age_group == '80+':
        new_row['Age'] = random.randint(80, 99)

    # apply "No Finding" constraints
    new_row = apply_no_finding_constraints(new_row)

    # handle "AP/PA" and "Frontal/Lateral" relationship
    if new_row['Frontal/Lateral'] == 'Lateral':
        new_row['AP/PA'] = '0'
    else:
        new_row['AP/PA'] = random.choice(['AP', 'PA'])

    return new_row


input_df = pd.read_csv(input_csv_path)


disease_columns = [
    'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion',
    'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
    'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
]


generation_ratio = 0.25  
rows_to_generate = int(generation_ratio * len(input_df))
patient_id_counter = 400000  # Initial synthetic patient ID, pretty arbitrary, just make sure it dont overlap with any existing entries

# add a stratify column to the dataframe
input_df['stratify_col'] = input_df[disease_columns + ['Sex', 'Age Group']].apply(
    lambda row: '_'.join(row.values.astype(str)), axis=1
)

# calculate the normalized distribution of the stratify_col
distribution = input_df['stratify_col'].value_counts(normalize=True)

new_rows = []

# new rows
for _ in tqdm(range(rows_to_generate), desc="Generating new rows"):
    # sample a stratified row based on the distribution
    stratified_sample = distribution.sample(n=1, weights=distribution).index[0]
    
    # select a random row from the input dataframe matching the sampled stratified condition
    sampled_row = input_df[input_df['stratify_col'] == stratified_sample].sample(n=1).iloc[0]

    # generate a new synthetic row
    new_sample = generate_new_sample(sampled_row, patient_id_counter)
    new_rows.append(new_sample)

    # Increment the patient ID counter
    patient_id_counter += 1

# new df for new rows
new_rows_df = pd.DataFrame(new_rows)

# save the generated rows to a CSV file
output_filename = os.path.join(output_directory, "additional_samples_sampled_from_test_part_3.csv")
new_rows_df.to_csv(output_filename, index=False)

print(f"Generated {rows_to_generate} new rows and saved to {output_filename}.")
