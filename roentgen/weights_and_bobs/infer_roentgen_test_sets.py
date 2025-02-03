# Creates test sets of images according to the test set labels seen by the downstream classifier.
# i.e., downstream classifier was tested on real world dataset A, this script will generate  datasets with the same labels as dataset A but the images will be made by your roentgen weights.
# csv file per dataset generated must match exactly the underdiagnosis bias code csv format for testing.
# so all you need to do:
# read test set csv
# in loop load different model checkpoints per iteration
# each model checkpoint will generate images according to the test set labels
# save the images and respective new test set csv
# the main loop runs dataset generation on the model checkpoints from during training, after which a final generation is done with the latest/final model weights fromafter training was finished

import os
import pandas as pd
from diffusers import DiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel
import torch
from PIL import Image
from tqdm import tqdm

# generate text prompt based on conditions
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

# Path to the directory where datasets will be saved
#output_base_dir = "/workspace/my_auxiliary_persistent/roentgen_datasets/hugging_out_5_part4_ground_truth"
#output_base_dir = "/workspace/my_auxiliary_persistent/retrain_roentgen/full_test_set_for_metric_assessment"
#output_base_dir = "/workspace/my_auxiliary_persistent/retrain_roentgen/full_gt_set_for_metric_assessment"
#output_base_dir = "/workspace/my_auxiliary_persistent/upsampled_test_set/imgs_2"
#output_base_dir = "/workspace/my_auxiliary_persistent/retrain_roentgen/train_on_test/additional_images/the_images"
output_base_dir = "/workspace/my_auxiliary_persistent/retrain_roentgen/train_on_test/additional_images/the_images_3"

# path to the CSV file
#csv_path = "/workspace/persistent/code/roentgen/weights_and_bobs/diffusion_splits/processed_valid_with_projection_corr_rot.csv"
#csv_path = "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/ground_truth.csv" # 1st one I used to generate
#csv_path = "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/test.csv" # trying 2 get more with cpu
#csv_path = "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/test_40-80.csv" # trying 2 get more with cpu
#csv_path = "/workspace/my_auxiliary_persistent/retrain_roentgen/train_on_test/additional_images/additional_samples_sampled_from_test.csv"
csv_path = "/workspace/my_auxiliary_persistent/retrain_roentgen/train_on_test/additional_images/additional_samples_sampled_from_test_part_3.csv"


model_checkpoints = [ 
    #"checkpoint-19500",
    #"checkpoint-25500",
    #"checkpoint-30500",
    #"checkpoint-34500",
    #"checkpoint-1000",
    #"checkpoint-4500",
    #"checkpoint-8000",
    #"checkpoint-14500",
    #"checkpoint-19000",
    #"checkpoint-36500",
    "checkpoint-44000", # retrained roentgen on tesst set, other checkpoints maybe referred to the first wrongly trained roentgen on validation.
]

df = pd.read_csv(csv_path)


os.makedirs(output_base_dir, exist_ok=True)


#base_model_path = "/workspace/persistent/code/roentgen/weights_and_bobs/hugging_out_5"
base_model_path = "/workspace/my_auxiliary_persistent/retrain_roentgen/finetuneing"

# max imgs per checkpooinit
max_images = 1000 # 4000 og 


for checkpoint in tqdm(model_checkpoints):
    print(f"Processing {checkpoint}...")
    #checkpoint_path = f"/workspace/persistent/code/roentgen/weights_and_bobs/hugging_out_5/{checkpoint}"
    checkpoint_path = f"/workspace/my_auxiliary_persistent/retrain_roentgen/finetuneing/{checkpoint}"

   
    unet = UNet2DConditionModel.from_pretrained(checkpoint_path, subfolder="unet")
    text_encoder = CLIPTextModel.from_pretrained(checkpoint_path, subfolder="text_encoder")
    
  
    pipeline = DiffusionPipeline.from_pretrained(base_model_path, unet=unet, text_encoder=text_encoder, safety_checker=None).to("cuda")
    #pipeline = DiffusionPipeline.from_pretrained(base_model_path, unet=unet, text_encoder=text_encoder, safety_checker=None).to("cpu")

 
    output_dir = os.path.join(output_base_dir, checkpoint)
    
    num_pics_thus_far = 0
    
    halfway_point = len(df) // 2  # Use integer division to get the midpoint
    
    #for idx, row in tqdm(df.iloc[:halfway_point].iterrows(), total=halfway_point, desc=f"Generating images for {checkpoint} (first half only)"):
    for idx, row in tqdm(df.iterrows(), total=len(df)):
    #for num_pics, (idx, row) in tqdm(enumerate(df.iterrows(), total=len(df))):# no work
        #if num_pics_thus_far >= max_images:
        #    break
        # Generate text prompt from CSV row
        
        # NEW, CHECKING IF IMAGE ALREADY EXISTS
        img_relative_path = row['Path']
        img_output_path = os.path.join(output_dir, img_relative_path)
        
        # Check if the image already exists
        if os.path.exists(img_output_path):
            print(f"Image already exists at {img_output_path}, skipping...")
            continue  # Skip to the next image if it already exists

        prompt = generate_text_prompt(row)

        # generate image
        out = pipeline(prompt, num_inference_steps=75, height=512, width=512, guidance_scale=4)
        img = out.images[0]

        # Construct output path
        # ORIGINAL, BEFORE I ADDED CHECKING IF IMAGE ALREADY EXISTS!!!
        #img_relative_path = row['Path']
        #img_output_path = os.path.join(output_dir, img_relative_path)

        # Ensure the directory exists
        os.makedirs(os.path.dirname(img_output_path), exist_ok=True)


        img.save(img_output_path)
        #print(f"Image saved to {img_output_path}")

        num_pics_thus_far += 1

print("image generation from earlier checkpoints complete, doing final generation round with final/latest model weights")
'''
# Final dataset generation with latest model weights
# Path to the latest model weights
latest_model_path = "/workspace/persistent/code/roentgen/weights_and_bobs/hugging_out_5"
pipeline = DiffusionPipeline.from_pretrained(latest_model_path, safety_checker=None).to("cuda")
#pipeline = DiffusionPipeline.from_pretrained(latest_model_path, safety_checker=None).to("cpu")


latest_output_dir = os.path.join(output_base_dir, "latest_model")


os.makedirs(latest_output_dir, exist_ok=True)

num_pics_thus_far = 0
# Loop through each row in the dataframe
for idx, row in tqdm(df.iterrows(), total=len(df)):
    #if num_pics_thus_far >= max_images:
    #    break


    prompt = generate_text_prompt(row)

    out = pipeline(prompt, num_inference_steps=75, height=512, width=512, guidance_scale=4)
    img = out.images[0]


    img_relative_path = row["Path"]
    img_output_path = os.path.join(latest_output_dir, img_relative_path)

    os.makedirs(os.path.dirname(img_output_path), exist_ok=True)


    img.save(img_output_path)
    #print(f"Image saved to {img_output_path}")
    
    num_pics_thus_far += 1

print("Image generation complete.")
'''



