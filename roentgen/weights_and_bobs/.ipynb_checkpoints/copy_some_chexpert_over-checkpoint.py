import pandas as pd
import os
import shutil
import random


def generate_text_prompt(row):
    prompt = f"{row['Sex']}, age group {row['Age Group']}, view {row['Frontal/Lateral']}"
    conditions = [
        "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema",
        "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
        "Pleural Other", "Fracture", "Support Devices", "No Finding"
    ]
    for condition in conditions:
        if row[condition] == 1:
            prompt += f", {condition}"
    return prompt.strip(", ")


csv_path = "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/train.csv"
source_prefix = "/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/"
destination_folder = "/workspace/persistent/code/roentgen/weights_and_bobs/sample_chexpert_images/"


data = pd.read_csv(csv_path)

# num of random images to sample
num_samples = 300  # Adjust this number as needed

sampled_data = data.sample(n=num_samples, random_state=42)

os.makedirs(destination_folder, exist_ok=True)

copy and rename imgs
for index, row in sampled_data.iterrows():
    source_path = os.path.join(source_prefix, row['Path'])
    if os.path.exists(source_path):
        text_prompt = generate_text_prompt(row)
        file_extension = os.path.splitext(source_path)[1]
        destination_path = os.path.join(destination_folder, f"{text_prompt}{file_extension}")
        shutil.copy(source_path, destination_path)

print("Images copied and renamed successfully.")

