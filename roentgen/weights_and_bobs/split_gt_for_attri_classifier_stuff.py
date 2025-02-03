import pandas as pd
# split ground truth csv for attribute classifier 

input_csv = "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_3/refit_for_attribute_classifier_ground_truth.csv"
output_dir = "/workspace/my_auxiliary_persistent/retrain_roentgen/attribute_classifier_part_2/gt_attr_splits/"


data = pd.read_csv(input_csv)

# Define the age groups and their corresponding labels
age_groups = {
    "0-20": (0, 20),
    "20-40": (20, 40),
    #"40-80": (40, 80),
    "80+": (80, float('inf'))
}

# Iterate over age groups
for age_label, (age_min, age_max) in age_groups.items():
    # Filter data for the current age group
    age_filtered = data[(data['Age'] >= age_min) & (data['Age'] <= age_max)]

    # Save the "all sexes" CSV file for the age group
    age_all_sexes_path = f"{output_dir}gt_{age_label}_split_all_sexes.csv"
    age_filtered.to_csv(age_all_sexes_path, index=False)

    # Further split by sex
    for sex in ['Male', 'Female']:
        sex_filtered = age_filtered[age_filtered['Sex'] == sex]

        # Save the CSV file for the specific sex and age group
        sex_path = f"{output_dir}gt_{age_label}_split_{sex.lower()}.csv"
        sex_filtered.to_csv(sex_path, index=False)

print("CSV files split and saved successfully.")
