import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim import AdamW
from PIL import Image
from diffusers import StableDiffusionPipeline
from torch.nn import MSELoss

#PATH_TO_IMAGES = "/home/kaspar/src/python_projects/master_thesis/natmed_stuff/data/images_reduced_arbitrary/"
PATH_TO_IMAGES = "/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/"
#CSV_FILE = "/home/kaspar/src/python_projects/master_thesis/my_roentgen/data/splits/test_reduced.csv"
CSV_FILE = "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/valid.csv"
#MODEL_PATH = "/home/kaspar/src/python_projects/master_thesis/my_roentgen/roentgen"
MODEL_PATH = "/workspace/persistent/code/roentgen/weights_and_bobs/roentgen"
#TUNED_MODEL_SAVE_PATH = "/home/kaspar/src/python_projects/master_thesis/my_roentgen/my_saved_weights"
TUNED_MODEL_SAVE_PATH = "/workspace/persistent/code/roentgen/weights_and_bobs/my_saved_weights"

# custom dataset class
class CustomChestXRayDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row["Path"])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        # Generate text prompt
        # TODO: ADD LATERAL/FRONTAL to text prompt. Also, OG paper guys didnt really do lateral images?
        prompt = f"{row['Sex']}, age group {row['Age Group']}"
        conditions = [
            "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema",
            "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
            "Pleural Other", "Fracture", "Support Devices", "No Finding"
        ]
        for condition in conditions:
            if row[condition] == 1:
                prompt += f", {condition}"

        return {"image": image, "prompt": prompt}

# init. dataset and dataloader
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

dataset = CustomChestXRayDataset(CSV_FILE, PATH_TO_IMAGES, transform=transform)
#dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define indices for training and validation using stratified split
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
train_idx, val_idx = next(splitter.split(dataset.data, dataset.data[['Sex', 'Age Group']]))

# Create data loaders for training and validation
train_dataset = torch.utils.data.Subset(dataset, train_idx)
val_dataset = torch.utils.data.Subset(dataset, val_idx)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)  # No need to shuffle validation data


# load pretrained RoentGen model
pipe = StableDiffusionPipeline.from_pretrained(MODEL_PATH)

# try freeze safety layer
#pipe.safety_checker = lambda imgs, _: (imgs, False)
pipe.safety_checker = None

# freeze VAE
for param in pipe.vae.parameters():
    param.requires_grad = False
# NOTE: MAYBE FREEZING THIS WAY IS CORRECT OR BETTER:
# vae.requires_grad_(False)

# optionally freeze some layers of the U-Net and text encoder
for param in pipe.unet.parameters():
    param.requires_grad = True  # Set to False to freeze specific layers

for param in pipe.text_encoder.parameters():
    param.requires_grad = True  # Set to False to freeze specific layers

#optimizer = AdamW(filter(lambda p: p.requires_grad, pipe.parameters()), lr=5e-5)
params_to_optimize = list(pipe.unet.parameters()) + list(pipe.text_encoder.parameters())
optimizer = AdamW(filter(lambda p: p.requires_grad, params_to_optimize), lr=5e-5)

# loss function for training
loss_criterion = MSELoss()

# training loop
num_epochs = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"
pipe.to(device)

'''
for epoch in range(num_epochs):
    for batch in dataloader:
        images = batch["image"].to(device)
        prompts = batch["prompt"]

        # Tokenize text
        #tokens = pipe.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)

        # Forward pass
        #outputs = pipe(prompt=prompts, images=images, return_loss=True)
        generated_images = pipe(prompt=prompts).images


        # convert generated images to tensors
        generated_tensors = []
        for img in generated_images:
            img_tensor = transforms.ToTensor()(img).to(device)
            img_tensor = transforms.Normalize([0.5], [0.5])(img_tensor)
            img_tensor.requires_grad = True
            generated_tensors.append(img_tensor)
        generated_tensors = torch.stack(generated_tensors)


        #loss = outputs.loss  # Loss is automatically provided
        loss = loss_criterion(generated_tensors, images)

        # bckward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")
'''

max_timesteps = 100

for epoch in range(num_epochs):
    pipe.train()
    for batch in train_datalaoder: #dataloader:
        images = batch["image"].to(device)
        prompts = batch["prompt"]

        # Tokenize prompts
        tokenized_prompts = pipe.tokenizer(prompts, padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids.to(device)

        # Encode images and prompts
        with torch.no_grad():
            latents = pipe.vae.encode(images).latent_dist.sample() * pipe.vae.config.scaling_factor
            text_embeddings = pipe.text_encoder(tokenized_prompts).last_hidden_state

        # Sample random noise to add to the latent space
        noise = torch.randn_like(latents).to(device)
        noisy_latents = latents + noise
        
        timesteps = torch.randint(1, max_timesteps, (1,), device=device).item()

        # Predict noise with the U-Net
        predicted_noise = pipe.unet(noisy_latents, timestep=timesteps, encoder_hidden_states=text_embeddings).sample

        # Compute the loss
        loss = loss_criterion(predicted_noise, noise)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

    pipe.eval()
    val_loss_total = 0.0
    with torch.no_grad():
        for batch in val_dataloader:
            images = batch["image"].to(device)
            prompts = batch["prompt"]

            tokenized_prompts = pipe.tokenizer(prompts, padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids.to(device)

            latents = pipe.vae.encode(images).latent_dist.sample() * pipe.vae.config.scaling_factor
            text_embeddings = pipe.text_encoder(tokenized_prompts).last_hidden_state

            noise = torch.randn_like(latents).to(device)
            noisy_latents = latents + noise
            
            timesteps = torch.randint(1, max_timesteps, (1,), device=device).item()

            predicted_noise = pipe.unet(noisy_latents, timestep=timesteps, encoder_hidden_states=text_embeddings).sample

            val_loss = loss_criterion(predicted_noise, noise)
            val_loss_total += val_loss.item()

    avg_val_loss = val_loss_total / len(val_dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {avg_val_loss}")

print("finetune training complete")
print("saving model and/or weights in: " + TUNED_MODEL_SAVE_PATH)
pipe.save_pretrained(TUNDED_MODEL_SAVE_PATH)
print("saved model and/or weights")



