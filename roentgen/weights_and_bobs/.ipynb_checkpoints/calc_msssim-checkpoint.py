import os
import torch
import torchvision.transforms as T
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from itertools import combinations
from tqdm import tqdm
import scipy.stats as st
import pandas as pd
from diffusers import DiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel

# generate text prompt based on conditions seen in og csv
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


transform = T.Compose([
    #T.Resize((256, 256)),
    T.ToTensor()
])

# msssim between 2 images
def calculate_ssim(img1, img2):
    img1 = np.array(img1)
    img2 = np.array(img2)
    ssim_value, _ = ssim(img1, img2, channel_axis=2, full=True, sigma=1.5, win_size=11)
    return ssim_value

# msssim for one set off generated images
def calculate_msssim_for_samples(samples):
    ssim_values = []
    # pairwise msssim for all combos
    for img1, img2 in combinations(samples, 2):
        ssim_values.append(calculate_ssim(img1, img2))
    return np.mean(ssim_values)

# generate n sets of imgs per prompt
def generate_images_for_prompt(pipeline, prompt, n=10, steps=75, height=512, width=512, guidance_scale=4):
    image_sets = []
    for _ in range(n):
        # Generate image
        out = pipeline(prompt, num_inference_steps=steps, height=height, width=width, guidance_scale=guidance_scale)
        img = out.images[0]
        image_sets.append(img)
    return image_sets

# calc msssim & conf. interval
def calculate_msssim_with_ci(pipeline, df, n_sets_per_prompt=10, m_iterations=100):
    msssim_scores = []

    #  m iterations of randomly selecting rows from the dataframe
    for _ in tqdm(range(m_iterations), total=m_iterations):
        # select random idx from dataframe (with replacement)
        random_idx = np.random.choice(df.index)
        row = df.loc[random_idx]
        
        #gen prompt and imgs
        prompt = generate_text_prompt(row)
        image_sets = generate_images_for_prompt(pipeline, prompt, n=n_sets_per_prompt)
        
        # calc mssim for dis prompt for this prompt
        msssim_score = calculate_msssim_for_samples(image_sets)
        msssim_scores.append(msssim_score)

    # mean MS-SSIM across all selected prompts
    mean_msssim = np.mean(msssim_scores)
    #  95% CI
    ci_low, ci_high = st.t.interval(0.95, len(msssim_scores) - 1, loc=mean_msssim, scale=st.sem(msssim_scores))
    return mean_msssim, (ci_low, ci_high)

# main for running alll
def evaluate_image_diversity(pipeline, df, n_sets_per_prompt=10, m_iterations=100):
    print("Starting MS-SSIM calculation...")
    mean_msssim, ci = calculate_msssim_with_ci(pipeline, df, n_sets_per_prompt=n_sets_per_prompt, m_iterations=m_iterations)
    print(f"Mean MS-SSIM: {mean_msssim:.4f} ± {ci[1] - mean_msssim:.4f}")
    return mean_msssim, ci


#csv_path = "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/ground_truth.csv"
csv_path = "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/test.csv"
df = pd.read_csv(csv_path)

# Load the custom UNet and text encoder
#checkpoint_path = f"/workspace/persistent/code/roentgen/weights_and_bobs/hugging_out_5/checkpoint-25500"
checkpoint_path = "/workspace/my_auxiliary_persistent/retrain_roentgen/finetuneing/checkpoint-44000"
#base_model_path = "/workspace/persistent/code/roentgen/weights_and_bobs/hugging_out_5"
base_model_path = "/workspace/my_auxiliary_persistent/retrain_roentgen/finetuneing"

unet = UNet2DConditionModel.from_pretrained(checkpoint_path, subfolder="unet")
text_encoder = CLIPTextModel.from_pretrained(checkpoint_path, subfolder="text_encoder")

# Load the pipeline with custom components
pipeline = DiffusionPipeline.from_pretrained(base_model_path, unet=unet, text_encoder=text_encoder, safety_checker=None).to("cuda")

run eval.
# NOTE: n_sets_per_prompt is actually the number of images per prompt to generate, rather than actually n sets per prompt!
mean_msssim, confidence_interval = evaluate_image_diversity(pipeline, df, n_sets_per_prompt=4, m_iterations=750) # 4, 750

#output_file_path = "/workspace/persistent/code/roentgen/weights_and_bobs/calc_msssim_output.txt"
output_file_path = "/workspace/my_auxiliary_persistent/retrain_roentgen/eval_metrics_final/msssim_out.txt"

#write otuput fto file
with open(output_file_path, 'a') as f:  # 'a' mode appends to the existing file
    f.write(f"Mean MS-SSIM: {mean_msssim:.4f}\n")
    f.write(f"95% Confidence Interval: [{confidence_interval[0]:.4f}, {confidence_interval[1]:.4f}]\n")
    f.write(f"± {confidence_interval[1] - mean_msssim:.4f}\n")
    f.write("\n")  # Add a blank line for separation between results



