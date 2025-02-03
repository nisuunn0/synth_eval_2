import os
import pandas as pd

def check_patient_overlap(csv_folder):
    #  list of CSV files in the folder
    csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]

    # dict to store patients and their corresponding csv files
    patient_files = {}

    for csv_file in csv_files:
        df = pd.read_csv(os.path.join(csv_folder, csv_file))

        df['Patient_ID'] = df['Path'].apply(lambda x: x.split('/')[1])

        for patient_id in df['Patient_ID'].unique():
            if patient_id not in patient_files:
                patient_files[patient_id] = []

            # append CSV file to list of files for this patient
            patient_files[patient_id].append(csv_file)

    # check for patients occurring in multiple csv files
    for patient_id, files in patient_files.items():
        if len(files) > 1:
            print(f"Patient {patient_id} occurs in multiple CSV files: {', '.join(files)}")


csv_folder = "./my_splits_3"


check_patient_overlap(csv_folder)

