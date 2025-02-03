import os
import pandas as pd
from diffusers import DiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel
from tqdm import tqdm

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

# base directory for saving images
#output_base_dir = "/workspace/my_auxiliary_persistent/upsampled_test_set/imgs/"
output_base_dir = "/workspace/my_auxiliary_persistent/retrain_roentgen/generated_test_sets/imgs/"
# Path to the directory containing the CSV files
#csv_files_dir = "/workspace/my_auxiliary_persistent/upsampled_test_set/csv_files/upsamples_to_add_to_test/"
csv_files_dir = "/workspace/my_auxiliary_persistent/upsampled_test_set/csv_files/upsamples_to_add_to_upsamples/"
# Checkpoint to load
#checkpoint = "checkpoint-25500"
checkpoint = "checkpoint-44000"

#base_model_path = "/workspace/persistent/code/roentgen/weights_and_bobs/hugging_out_5"
base_model_path = "/workspace/my_auxiliary_persistent/retrain_roentgen/finetuneing"

# load the model from the checkpoint
#checkpoint_path = f"/workspace/persistent/code/roentgen/weights_and_bobs/hugging_out_5/{checkpoint}"
checkpoint_path = f"/workspace/my_auxiliary_persistent/retrain_roentgen/finetuneing/{checkpoint}"
unet = UNet2DConditionModel.from_pretrained(checkpoint_path, subfolder="unet")
text_encoder = CLIPTextModel.from_pretrained(checkpoint_path, subfolder="text_encoder")
pipeline = DiffusionPipeline.from_pretrained(base_model_path, unet=unet, text_encoder=text_encoder, safety_checker=None).to("cuda")

# ensure output base directory exists
os.makedirs(output_base_dir, exist_ok=True)

# get a list of all CSV files in the specified directory
csv_files = [f for f in os.listdir(csv_files_dir) if f.endswith('.csv')]

# iterate over each CSV file
for csv_file in tqdm(csv_files):
    csv_path = os.path.join(csv_files_dir, csv_file)
    df = pd.read_csv(csv_path)

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # generate text prompt from CSV row
        prompt = generate_text_prompt(row)

        img_relative_path = row['Path']
        img_output_path = os.path.join(output_base_dir, img_relative_path)

        if os.path.exists(img_output_path):
            print(f"Image already exists at {img_output_path}, skipping...")
            continue  # Skip to the next image if it already exists

        # generate img
        out = pipeline(prompt, num_inference_steps=75, height=512, width=512, guidance_scale=4)
        img = out.images[0]

        # ensure the directory exists
        os.makedirs(os.path.dirname(img_output_path), exist_ok=True)


        img.save(img_output_path)

print("img generation complete for all specified CSV files.")

