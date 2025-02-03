# SAME CODE AS infer_roentgen_test_sets.py but instead of generating full source csv contents, this one samples based on stratification a subset of source csv file, so that quickly small samples per model checkpoint can be generated for quick evaluation
# in an attempt to determine the best diffusion model checkpoint

import os
import pandas as pd
from diffusers import DiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel
import torch
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

#generate text prompt based on conditions
def generate_text_prompt(row):
    prompt = f"{row['Sex']}, age group {row['Age Group']}, view {row['Frontal/Lateral']}"
    if row['AP/PA'] == "AP" or row['AP/PA'] == "PA":
        prompt += f", projection {row['AP/PA']}"
    conditions = [
        "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema",
        "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
        "Pleural Other", "Fracture", "Support Devices", "No Finding"
    ]
    for condition in conditions:
        if row[condition] == 1.0:
            prompt += f", {condition}"
    return prompt.strip(", ")


#csv_path = "/path/to/your/ground_truth.csv"
csv_path = "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/test.csv"
#output_base_dir = "/workspace/my_auxiliary_persistent/roentgen_datasets/hugging_out_5_part4_ground_truth"
output_base_dir = "/workspace/my_auxiliary_persistent/retrain_roentgen/test_checkpoints_images"
df = pd.read_csv(csv_path)


os.makedirs(output_base_dir, exist_ok=True)

# max num imgs per checkpoint
max_images = 800  
random_seed = 42

# sample `max_images` rows with stratification on key conditions
df_subset, _ = train_test_split(
    df,
    train_size=max_images,
    random_state=random_seed,
    stratify=df[["Sex", "Frontal/Lateral"] + ["No Finding", "Support Devices"]] #conditions]  # stratify on main conditions
)

model_checkpoints = [
    "checkpoint-75500",
    "checkpoint-73000",
    "checkpoint-68500",
    "checkpoint-60500",
    "checkpoint-54500",
    "checkpoint-44000",
    "checkpoint-39500",
    "checkpoint-33000",
    "checkpoint-21000",
    "checkpoint-12500",
    "checkpoint-36000",
    "checkpoint-64500",
]


#base_model_path = "/workspace/persistent/code/roentgen/weights_and_bobs/hugging_out_5"
base_model_path = "/workspace/my_auxiliary_persistent/retrain_roentgen/finetuneing"

for checkpoint in tqdm(model_checkpoints):
    print(f"Processing {checkpoint}...")
    checkpoint_path = f"{base_model_path}/{checkpoint}"

    # load the custom UNet and text encoder
    unet = UNet2DConditionModel.from_pretrained(checkpoint_path, subfolder="unet")
    text_encoder = CLIPTextModel.from_pretrained(checkpoint_path, subfolder="text_encoder")
    
    # load the pipeline with custom components
    pipeline = DiffusionPipeline.from_pretrained(base_model_path, unet=unet, text_encoder=text_encoder, safety_checker=None).to("cuda")

    # directory for current checkpoint
    output_dir = os.path.join(output_base_dir, checkpoint)
    os.makedirs(output_dir, exist_ok=True)

    # generate images for the selected subset of rows
    for idx, row in tqdm(df_subset.iterrows(), total=len(df_subset)):
        # Construct image output path
        img_relative_path = row['Path']
        img_output_path = os.path.join(output_dir, img_relative_path)

        # Check if image already exists
        if os.path.exists(img_output_path):
            print(f"Image already exists at {img_output_path}, skipping...")
            continue

        # generate text prompt and image
        prompt = generate_text_prompt(row)
        out = pipeline(prompt, num_inference_steps=75, height=512, width=512, guidance_scale=4)
        img = out.images[0]

        os.makedirs(os.path.dirname(img_output_path), exist_ok=True)
        img.save(img_output_path)

print("Image generation from earlier checkpoints complete.")

