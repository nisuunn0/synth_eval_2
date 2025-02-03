import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import os
import clip
from tqdm import tqdm
import numpy as np
from scipy.linalg import sqrtm

# Load DenseNet-121 model 
def load_xrv_model():
    CheckPointData = torch.load('/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_results_batch_size_192/checkpoint')  # Load your specific checkpoint
    model = CheckPointData['model']  # Extract the model from the checkpoint
    model.classifier = torch.nn.Identity()

    model.eval()  
    return model


class ImageDataset(Dataset):
    def __init__(self, csv_file, img_dir_prefix, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir_prefix = img_dir_prefix  # path prefix (real or generated)
        self.transform = transform
    
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        # prepend the specified directory (real or generated) to the path from CSV
        img_path = os.path.join(self.img_dir_prefix, self.data_frame.iloc[idx]['Path'])  
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, 0  # only need img return dummy

# FID helper functions 
def calculate_fid(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = np.sum(diff**2) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

def get_features(model, dataloader, device, dims, is_clip=False):
    features = np.zeros((len(dataloader.dataset), dims))
    idx = 0

    with torch.no_grad():
        for imgs, _ in tqdm(dataloader):
            imgs = imgs.to(device)

            if is_clip:
                # CLIP model uses encode_image directly
                output = model(imgs).cpu().numpy()
            else:
                # other models go through the normal process
                model.eval()  # Only call eval on non-CLIP models
                output = model(imgs).cpu().numpy()

            features[idx:idx + output.shape[0]] = output
            idx += output.shape[0]

    return features


def compute_statistics(features):
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma

########################### USE THIS IF YOU WANT TO USE A SUBSET OF THE CSV FILE #############################
def create_dataloaders(csv_file, img_dir_real, img_dir_generated, batch_size=32, n_images=None):
    # Load the CSV
    data_frame = pd.read_csv(csv_file)
    print(f"Original CSV length: {len(data_frame)}")  # Print original length

    # Filter for images that exist in both directories
    filtered_data_frame = data_frame[
        data_frame['Path'].apply(
            lambda x: os.path.exists(os.path.join(img_dir_real, x)) and 
                      os.path.exists(os.path.join(img_dir_generated, x))
        )
    ]
    print(f"Filtered CSV length: {len(filtered_data_frame)}")  # Print length after filtering

    # save the filtered CSV temporarily (for ImageDataset to load)
    filtered_csv_path = '/workspace/my_auxiliary_persistent/retrain_roentgen/temp_files_delet/temp_filtered.csv'
    filtered_data_frame.to_csv(filtered_csv_path, index=False)

    # use the filtered CSV to create datasets
    transform_inception = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_clip = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])

    # use the filtered CSV for both datasets
    real_dataset_inception = ImageDataset(csv_file=filtered_csv_path, img_dir_prefix=img_dir_real, transform=transform_inception)
    real_dataset_clip = ImageDataset(csv_file=filtered_csv_path, img_dir_prefix=img_dir_real, transform=transform_clip)
    
    generated_dataset_inception = ImageDataset(csv_file=filtered_csv_path, img_dir_prefix=img_dir_generated, transform=transform_inception)
    generated_dataset_clip = ImageDataset(csv_file=filtered_csv_path, img_dir_prefix=img_dir_generated, transform=transform_clip)

    # apply any optional image limit
    if n_images:
        real_dataset_inception = torch.utils.data.Subset(real_dataset_inception, list(range(n_images)))
        real_dataset_clip = torch.utils.data.Subset(real_dataset_clip, list(range(n_images)))
        generated_dataset_inception = torch.utils.data.Subset(generated_dataset_inception, list(range(n_images)))
        generated_dataset_clip = torch.utils.data.Subset(generated_dataset_clip, list(range(n_images)))

    # create dataloaders
    real_dataloader_inception = DataLoader(real_dataset_inception, batch_size=batch_size, shuffle=False)
    real_dataloader_clip = DataLoader(real_dataset_clip, batch_size=batch_size, shuffle=False)
    
    generated_dataloader_inception = DataLoader(generated_dataset_inception, batch_size=batch_size, shuffle=False)
    generated_dataloader_clip = DataLoader(generated_dataset_clip, batch_size=batch_size, shuffle=False)

    return real_dataloader_inception, generated_dataloader_inception, real_dataloader_clip, generated_dataloader_clip





'''
# DEFAULT create_dataloaders function
#create the dataloaders with separate transforms for CLIP and Inception
def create_dataloaders(csv_file, img_dir_real, img_dir_generated, batch_size=32, n_images=None):
    # Define the necessary transformations for InceptionV3 (299x299)
    transform_inception = transforms.Compose([
        transforms.Resize((299, 299)),  # Size for InceptionV3
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])

    # transformations for CLIP (224x224)
    transform_clip = transforms.Compose([
        transforms.Resize((224, 224)),  # Size for CLIP
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])  # CLIP normalization
    ])

    # For the real and generated images (same CSV, different prefixes)
    real_dataset_inception = ImageDataset(csv_file=csv_file, img_dir_prefix=img_dir_real, transform=transform_inception)
    real_dataset_clip = ImageDataset(csv_file=csv_file, img_dir_prefix=img_dir_real, transform=transform_clip)
    
    generated_dataset_inception = ImageDataset(csv_file=csv_file, img_dir_prefix=img_dir_generated, transform=transform_inception)
    generated_dataset_clip = ImageDataset(csv_file=csv_file, img_dir_prefix=img_dir_generated, transform=transform_clip)

    if n_images:  # Limit the number of images for testing
        real_dataset_inception = torch.utils.data.Subset(real_dataset_inception, list(range(n_images)))
        real_dataset_clip = torch.utils.data.Subset(real_dataset_clip, list(range(n_images)))
        generated_dataset_inception = torch.utils.data.Subset(generated_dataset_inception, list(range(n_images)))
        generated_dataset_clip = torch.utils.data.Subset(generated_dataset_clip, list(range(n_images)))

    real_dataloader_inception = DataLoader(real_dataset_inception, batch_size=batch_size, shuffle=False)
    real_dataloader_clip = DataLoader(real_dataset_clip, batch_size=batch_size, shuffle=False)
    
    generated_dataloader_inception = DataLoader(generated_dataset_inception, batch_size=batch_size, shuffle=False)
    generated_dataloader_clip = DataLoader(generated_dataset_clip, batch_size=batch_size, shuffle=False)

    return real_dataloader_inception, generated_dataloader_inception, real_dataloader_clip, generated_dataloader_clip
'''

# FID calculation for all models (InceptionV3, CLIP-ViT-B/32, XRV-DenseNet-121)
def calculate_fid_for_all_models(generated_dataloader_inception, real_dataloader_inception,
                                 generated_dataloader_clip, real_dataloader_clip, 
                                 xrv_model, inception_model, clip_model, device):
    # InceptionV3
    inception_features_real = get_features(inception_model, real_dataloader_inception, device, 2048)
    inception_features_generated = get_features(inception_model, generated_dataloader_inception, device, 2048)
    mu_inception_real, sigma_inception_real = compute_statistics(inception_features_real)
    mu_inception_generated, sigma_inception_generated = compute_statistics(inception_features_generated)
    fid_inception = calculate_fid(mu_inception_real, sigma_inception_real, mu_inception_generated, sigma_inception_generated)

    # CLIP-ViT-B/32
    clip_features_real = get_features(clip_model.encode_image, real_dataloader_clip, device, 512, is_clip=True)
    clip_features_generated = get_features(clip_model.encode_image, generated_dataloader_clip, device, 512, is_clip=True)
    mu_clip_real, sigma_clip_real = compute_statistics(clip_features_real)
    mu_clip_generated, sigma_clip_generated = compute_statistics(clip_features_generated)
    fid_clip = calculate_fid(mu_clip_real, sigma_clip_real, mu_clip_generated, sigma_clip_generated)

    # XRV DenseNet-121
    xrv_features_real = get_features(xrv_model, real_dataloader_inception, device, 1024)  # Reusing Inception dataloaders
    xrv_features_generated = get_features(xrv_model, generated_dataloader_inception, device, 1024)
    mu_xrv_real, sigma_xrv_real = compute_statistics(xrv_features_real)
    mu_xrv_generated, sigma_xrv_generated = compute_statistics(xrv_features_generated)
    fid_xrv = calculate_fid(mu_xrv_real, sigma_xrv_real, mu_xrv_generated, sigma_xrv_generated)

    return fid_inception, fid_clip, fid_xrv

# 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: " + str(device))

# Load models
inception_model = models.inception_v3(pretrained=True, transform_input=False).to(device)
inception_model.fc = torch.nn.Identity()  # remove FC for feature extraction
xrv_model = load_xrv_model().to(device)  # your pre-trained model
clip_model, preprocess = clip.load("ViT-B/32", device=device)


#csv_file = '/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/ground_truth.csv'
csv_file = "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/test.csv"
real_image_dir = '/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/'
#generated_image_dir = '/workspace/my_auxiliary_persistent/roentgen_datasets/hugging_out_5_part4_ground_truth/checkpoint-25500/'
#generated_image_dir = "/workspace/my_auxiliary_persistent/retrain_roentgen/generated_test_sets/imgs/"
generated_image_dir = "/workspace/my_auxiliary_persistent/retrain_roentgen/full_test_set_for_metric_assessment/checkpoint-44000/"

# limit the dataloaders to use only 100 images for testing
n_images = None #100
real_dataloader_inception, generated_dataloader_inception, real_dataloader_clip, generated_dataloader_clip = create_dataloaders(
    csv_file, real_image_dir, generated_image_dir, n_images=n_images)

#  FID for all models
fid_inception, fid_clip, fid_xrv = calculate_fid_for_all_models(
    generated_dataloader_inception, real_dataloader_inception, 
    generated_dataloader_clip, real_dataloader_clip,
    xrv_model, inception_model, clip_model, device)

print(f"FID InceptionV3: {fid_inception}")
print(f"FID CLIP-ViT-B/32: {fid_clip}")
print(f"FID DenseNet-121 (XRV): {fid_xrv}")

# save FID scores to a text file
output_file_path = "/workspace/my_auxiliary_persistent/retrain_roentgen/eval_metrics_final/three_fids.txt"
with open(output_file_path, 'w') as file:
    file.write(f"FID InceptionV3: {fid_inception}\n")
    file.write(f"FID CLIP-ViT-B/32: {fid_clip}\n")
    file.write(f"FID DenseNet-121 (XRV): {fid_xrv}\n")

print("FID results saved to:", output_file_path)

# RESULTS for roentgen finetuned on validation set (1st model)
#FID InceptionV3: 41.87245568468953
#FID CLIP-ViT-B/32: 4.0131166593425105
#FID DenseNet-121 (XRV): 79.22275870367724


