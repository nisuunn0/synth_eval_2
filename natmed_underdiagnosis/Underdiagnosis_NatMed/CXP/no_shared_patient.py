import os
import random
import pandas as pd
import multiprocessing
from tqdm import tqdm

# fodler containing csvs
folder_path = "./my_splits"


csv_files = ["ground_truth.csv", "test.csv", "train.csv", "valid.csv"]


dataframes = {}

s
for file in csv_files:
    file_path = os.path.join(folder_path, file)
    dataframes[file] = pd.read_csv(file_path)

# combnee all patient IDs from all dataframes
print("Combining patient IDs...")
all_patients = set()
for df in tqdm(dataframes.values(), desc="processing CSV files"):
    all_patients.update(df['Path'].apply(lambda x: [part for part in x.split('/') if part.startswith('patient')][0]))

def compute_occurrences(patient):
    occurrences = [file for file, df in dataframes.items() if any(df['Path'].str.contains(f"/{patient}/"))]
    return patient, occurrences


print("precomputing occurrences for each patient...")
with multiprocessing.Pool() as pool:
    results = []
    for result in tqdm(pool.imap_unordered(compute_occurrences, all_patients), total=len(all_patients), desc="precomputing occurrences"):
        results.append(result)

# construct patient_occurrences dictionary from results
patient_occurrences = {patient: occurrences for patient, occurrences in results}

# Randomly assign destinations for each patient
print("anssigning patient destinations...")
patient_destinations = {}
for patient in tqdm(all_patients, desc="Assigning destinations"):
    occurrences = patient_occurrences[patient]
    if len(occurrences) > 1:
        destination = random.choice(occurrences)
        patient_destinations[patient] = destination

print("Moving patients to assigned destinations...")
for file, df in tqdm(dataframes.items(), desc="Moving patients"):
    for patient, destination in patient_destinations.items():
        if destination == file:
            continue
        if destination:
            dataframes[destination] = pd.concat([dataframes[destination], df[df['Path'].str.contains(f"/{patient}/")]])
            dataframes[file] = df[~df['Path'].str.contains(f"/{patient}/")]

print("Saving modified dataframes...")
for file, df in tqdm(dataframes.items(), desc="Saving CSV files"):
    df.to_csv(os.path.join(folder_path, file), index=False)

print("Process completed successfully.")

