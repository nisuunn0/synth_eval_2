################## purpose of script: compute msssim for multiple different roentgen checkpoints from finetuneing

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

# roentgen model checkpoints to investigate
checkpoint_dirs = [
    "/workspace/my_auxiliary_persistent/retrain_roentgen/finetuneing/checkpoint-75500/", 
    "/workspace/my_auxiliary_persistent/retrain_roentgen/finetuneing/checkpoint-73000/", 
    "/workspace/my_auxiliary_persistent/retrain_roentgen/finetuneing/checkpoint-68500/", 
    "/workspace/my_auxiliary_persistent/retrain_roentgen/finetuneing/checkpoint-60500/", 
    "/workspace/my_auxiliary_persistent/retrain_roentgen/finetuneing/checkpoint-54500/", 
    "/workspace/my_auxiliary_persistent/retrain_roentgen/finetuneing/checkpoint-44000/", 
    "/workspace/my_auxiliary_persistent/retrain_roentgen/finetuneing/checkpoint-39500/", 
    "/workspace/my_auxiliary_persistent/retrain_roentgen/finetuneing/checkpoint-33000/", 
    "/workspace/my_auxiliary_persistent/retrain_roentgen/finetuneing/checkpoint-21000/",
    "/workspace/my_auxiliary_persistent/retrain_roentgen/finetuneing/checkpoint-12500/",
    "/workspace/my_auxiliary_persistent/retrain_roentgen/finetuneing/checkpoint-36000/",
    "/workspace/my_auxiliary_persistent/retrain_roentgen/finetuneing/checkpoint-64500/",
]

# base model path
base_model_path = "/workspace/my_auxiliary_persistent/retrain_roentgen/finetuneing"

# generate text prompt based on conditions as seen in source dataset csv files
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

# to tensor
transform = T.Compose([
    T.ToTensor()
])

# compute MS-SSIM between two images
def calculate_ssim(img1, img2):
    img1 = np.array(img1)
    img2 = np.array(img2)
    ssim_value, _ = ssim(img1, img2, channel_axis=2, full=True, sigma=1.5, win_size=11)
    return ssim_value

# compute msssim for one set of generated samples
def calculate_msssim_for_samples(samples):
    ssim_values = []
    for img1, img2 in combinations(samples, 2):
        ssim_values.append(calculate_ssim(img1, img2))
    return np.mean(ssim_values)

# generate n sets of images for each prompt
def generate_images_for_prompt(pipeline, prompt, n=10, steps=75, height=512, width=512, guidance_scale=4):
    image_sets = []
    for _ in range(n):
        out = pipeline(prompt, num_inference_steps=steps, height=height, width=width, guidance_scale=guidance_scale)
        img = out.images[0]
        image_sets.append(img)
    return image_sets

# calculate MS-SSIM and confidence interval
def calculate_msssim_with_ci(pipeline, df, n_sets_per_prompt=10, m_iterations=100):
    msssim_scores = []
    for _ in tqdm(range(m_iterations), total=m_iterations):
        random_idx = np.random.choice(df.index)
        row = df.loc[random_idx]
        
        prompt = generate_text_prompt(row)
        image_sets = generate_images_for_prompt(pipeline, prompt, n=n_sets_per_prompt)
        
        msssim_score = calculate_msssim_for_samples(image_sets)
        msssim_scores.append(msssim_score)

    mean_msssim = np.mean(msssim_scores)
    ci_low, ci_high = st.t.interval(0.95, len(msssim_scores) - 1, loc=mean_msssim, scale=st.sem(msssim_scores))
    return mean_msssim, (ci_low, ci_high)

# main function to run MS-SSIM eval. for each checkpoint
def evaluate_image_diversity_for_checkpoints(checkpoint_dirs, df, n_sets_per_prompt=10, m_iterations=100):
    output_file_path = "/workspace/my_auxiliary_persistent/retrain_roentgen/eval_checkpoints/quick_eval_multi_checkpoint_msssim.txt"

    with open(output_file_path, 'a') as f:
        for checkpoint_path in checkpoint_dirs:
            print(f"Evaluating checkpoint: {checkpoint_path}")
            f.write(f"Evaluating checkpoint: {checkpoint_path}\n")
            
            unet = UNet2DConditionModel.from_pretrained(checkpoint_path, subfolder="unet")
            text_encoder = CLIPTextModel.from_pretrained(checkpoint_path, subfolder="text_encoder")
            
            pipeline = DiffusionPipeline.from_pretrained(base_model_path, unet=unet, text_encoder=text_encoder, safety_checker=None).to("cuda")
            
            mean_msssim, ci = calculate_msssim_with_ci(pipeline, df, n_sets_per_prompt=n_sets_per_prompt, m_iterations=m_iterations)
            ci_range = ci[1] - mean_msssim
            
            print(f"Mean MS-SSIM for {checkpoint_path}: {mean_msssim:.4f} ± {ci_range:.4f}")
            f.write(f"Mean MS-SSIM: {mean_msssim:.4f} ± {ci_range:.4f}\n")
            f.write(f"95% Confidence Interval: [{ci[0]:.4f}, {ci[1]:.4f}]\n\n")

csv_path = "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/test.csv"
df = pd.read_csv(csv_path)

# run eval for all checkpoints
evaluate_image_diversity_for_checkpoints(checkpoint_dirs, df, n_sets_per_prompt=4, m_iterations=140)

