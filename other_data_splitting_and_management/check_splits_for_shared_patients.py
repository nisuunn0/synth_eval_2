import os
import pandas as pd
import re

def extract_patient_id(path):
    match = re.search(r"patient\d+", path)
    return match.group(0) if match else None

def check_patient_overlap(csv_folder):
    # get list of csv files in folder
    csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]
    #csv_files = ["test.csv","train.csv", "valid.csv", "ground_truth.csv"]

    # create a dict to store patients and their corresponding csv files
    patient_files = {}

    for csv_file in csv_files:
        print("reading some csv  stuff")
        # read csv
        df = pd.read_csv(os.path.join(csv_folder, csv_file))

        # get em patient IDs from file paths
        df['Patient_ID'] = df['Path'].apply(extract_patient_id)

        for patient_id in df['Patient_ID'].unique():
            if patient_id not in patient_files:
                patient_files[patient_id] = []

            # append CSV file to list of files for this patient
            print("appending csv file to list of files for this patient")
            patient_files[patient_id].append(csv_file)

    # check for patients occurring in multiple CSV files
    for patient_id, files in patient_files.items():
        print("checking")
        if len(files) > 1:
            print(f"patient {patient_id} occurs in multiple csv files: {', '.join(files)}")

# folder containing the csv files
#csv_folder = "./my_splits"
csv_folder = "/home/kaspar/src/master_thesis/results_generative/diffusion_set_downstream_results_september/"

# check for patient overlap among splits
check_patient_overlap(csv_folder)

