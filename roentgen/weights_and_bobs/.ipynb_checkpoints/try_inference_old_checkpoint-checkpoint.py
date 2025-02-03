# inference with code from train_text_to_image.py from huggingface code, same as try_inference_2.py but with custom unet checkpoint loading
from diffusers import DiffusionPipeline, UNet2DConditionModel
import torch
import os
import pandas as pd

# og roentgen weights before finetuning
#model_path = "/workspace/persistent/code/roentgen/weights_and_bobs/roentgen"
# your finetuned
model_path = "/workspace/persistent/code/roentgen/weights_and_bobs/hugging_out"
unet_path = model_path + "/checkpoint-10000/unet"
csv_path = "/workspace/persistent/code/roentgen/weights_and_bobs/use_these_csvs_for_finetune_man/processed_valid.csv"

device = "cpu" #"cuda"
# default pipe loading
#pipeline = DiffusionPipeline.from_pretrained(model_path, ).to(device) #torch_dtype=torch.float16).to(device)
# load same model but unet from some other checkpoint
unet = UNet2DConditionModel.from_pretrained(unet_path, )#torch_dtype=torch.float16)
pipeline = DiffusionPipeline.from_pretrained(model_path, unet=unet).to(device) #torch_dtype=torch.float16).to(device)


df = pd.read_csv(csv_path)

# num of prompts to read from the CSV
num_prompts = 15 


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
    out = pipeline(prompt, num_inference_steps=50, height=512, width=512, guidance_scale=4) # roentgen code inference
    #out = pipeline(prompt) # Hugging face training script inference
    img = out.images[0]
    img_filename = f"{prompt.replace(' ', '_')}.png"
    img_path = os.path.join(output_dir, img_filename)
    img.save(img_path)
    print(f"Image saved to {img_path}")


