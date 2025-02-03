# inference with code from train_text_to_image.py from huggingface code
from diffusers import DiffusionPipeline
import torch
import os
import pandas as pd

# og roentgen weights before finetuning
#model_path = "/workspace/persistent/code/roentgen/weights_and_bobs/roentgen"
# your finetuned
model_path = "/workspace/persistent/code/roentgen/weights_and_bobs/hugging_out_4"
#csv_path = "/workspace/persistent/code/roentgen/weights_and_bobs/use_these_csvs_for_finetune_man/processed_valid.csv"
csv_path = "/workspace/persistent/code/roentgen/weights_and_bobs/diffusion_splits/processed_valid_with_projection_corr_rot.csv"

device = "cuda"
pipeline = DiffusionPipeline.from_pretrained(model_path, ).to(device) #torch_dtype=torch.float16).to(device)

# Load the CSV file
df = pd.read_csv(csv_path)

# num prompts to read from csvb
num_prompts = 100 #250  # Change this value to read more or fewer prompts


#prompts = df['text'].head(num_prompts).tolist()
# random promptsfrin csv file
prompts = df['text'].sample(n=num_prompts).tolist()

#prompts = ["Female, age group 40-80, view Frontal, Lung Opacity, Pleural Effusion, Support Devices"]

#image = pipeline(prompt).images[0]
#image.save("my_image.png")






output_dir = "/workspace/persistent/code/roentgen/weights_and_bobs/out"

os.makedirs(output_dir, exist_ok=True)

# generate and save images for each prompt
for prompt in prompts:
    out = pipeline(prompt, num_inference_steps=75, height=512, width=512, guidance_scale=4) # roentgen code inference
    #out = pipeline(prompt) # Hugging face training script inference
    img = out.images[0]
    img_filename = f"{prompt.replace(' ', '_')}.png"
    img_path = os.path.join(output_dir, img_filename)
    img.save(img_path)
    print(f"Image saved to {img_path}")


