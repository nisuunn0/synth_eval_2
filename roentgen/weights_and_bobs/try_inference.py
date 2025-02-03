from diffusers import StableDiffusionPipeline
import torch
import os

def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))

force_cudnn_initialization()

#model_path = "/workspace/persistent/code/roentgen/weights_and_bobs/roentgen"
model_path = "/workspace/persistent/code/roentgen/weights_and_bobs/my_saved_weights_intermediate"
device = "cuda"  # or 'cpu' if you do not have a GPU

pipe = StableDiffusionPipeline.from_pretrained(model_path).to(torch.float32).to(device)


# print the architecture/components of the model
print("Feature Extractor:")
print(pipe.feature_extractor)
print("\nSafety Checker:")
print(pipe.safety_checker)
print("\nScheduler:")
print(pipe.scheduler)
print("\nText Encoder:")
print(pipe.text_encoder)
print("\nTokenizer:")
print(pipe.tokenizer)
print("\nUNet:")
print(pipe.unet)
print("\nVAE:")
print(pipe.vae)

prompts = [
    #"big right-sided pleural effusion",
    #"small left-sided pleural effusion",
    #"cardiomegaly with pulmonary edema"
    #"female aged 0-20",
    #"male aged 40-80",
    #"female aged 80+",
    #"broken ribs",
    #"cracked ribs",
    #"bruised ribs",
    "Female, 80+, Frontal, Support Devices",
    "Male, 0-20, Frontal, Consolidation",
    "Male, 0-20, Frontal, No Finding"
]

#output_dir = "/home/kaspar/src/python_projects/master_thesis/my_roentgen/out"
output_dir = "/workspace/persistent/code/roentgen/weights_and_bobs/out"

os.makedirs(output_dir, exist_ok=True)

# generate and save images for each prompt
for prompt in prompts:
    out = pipe(prompt, num_inference_steps=75, height=512, width=512, guidance_scale=4)
    img = out.images[0]
    img_filename = f"{prompt.replace(' ', '_')}.png"
    img_path = os.path.join(output_dir, img_filename)
    img.save(img_path)
    print(f"Image saved to {img_path}")

print("All images have been generated and saved.")




