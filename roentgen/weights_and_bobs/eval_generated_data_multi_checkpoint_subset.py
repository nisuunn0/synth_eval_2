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

# oad DenseNet-121 model
def load_xrv_model():
    CheckPointData = torch.load('/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_results_batch_size_192/checkpoint')
    model = CheckPointData['model']  # Extract the model from the checkpoint
    model.classifier = torch.nn.Identity()
    model.eval()
    return model

# dataset class to load images from CSV paths (with real/generated path prefix)
class ImageDataset(Dataset):
    def __init__(self, csv_file, img_dir_prefix, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir_prefix = img_dir_prefix
        self.transform = transform
    
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir_prefix, self.data_frame.iloc[idx]['Path'])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, 0  # return dummy label as we only need images

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
                output = model(imgs).cpu().numpy()
            else:
                model.eval()
                output = model(imgs).cpu().numpy()

            features[idx:idx + output.shape[0]] = output
            idx += output.shape[0]

    return features

def compute_statistics(features):
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma

# create the dataloaders, filtering images not in generated directory
def create_dataloaders(csv_file, img_dir_real, img_dir_generated, batch_size=32, n_images=None):
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

    # Filter csv to only include images present in the generated directory
    data_frame = pd.read_csv(csv_file)
    
    print(f"Initial number of images in CSV: {len(data_frame)}")
    
    data_frame['ImagePath'] = data_frame['Path'].apply(lambda x: os.path.join(img_dir_generated, x))
    data_frame = data_frame[data_frame['ImagePath'].apply(os.path.exists)]
    
    print(f"Number of images in CSV after filtering for existence in generated directory: {len(data_frame)}")
    
    filtered_csv = data_frame.drop(columns=['ImagePath'])

    # save the filtered CSV temporarily (for ImageDataset to load)
    filtered_csv.to_csv('/workspace/my_auxiliary_persistent/retrain_roentgen/eval_checkpoints/temp_delet/filtered_csv.csv', index=False)

    # create datasets for real and generated images
    #real_dataset_inception = ImageDataset(csv_file='/workspace/my_auxiliary_persistent/retrain_roentgen/eval_checkpoints/temp_delet/filtered_csv.csv', img_dir_prefix=img_dir_real, transform=transform_inception)
    #real_dataset_clip = ImageDataset(csv_file='/workspace/my_auxiliary_persistent/retrain_roentgen/eval_checkpoints/temp_delet/filtered_csv.csv', img_dir_prefix=img_dir_real, transform=transform_clip)
    real_dataset_inception = ImageDataset(csv_file='/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/test.csv', img_dir_prefix=img_dir_real, transform=transform_inception)
    real_dataset_clip = ImageDataset(csv_file='/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/test.csv', img_dir_prefix=img_dir_real, transform=transform_clip)
    #generated_dataset_inception = ImageDataset(csv_file='/workspace/my_auxiliary_persistent/retrain_roentgen/eval_checkpoints/temp_delet/filtered_csv.csv', img_dir_prefix=img_dir_generated, transform=transform_inception)
    #generated_dataset_clip = ImageDataset(csv_file='/workspace/my_auxiliary_persistent/retrain_roentgen/eval_checkpoints/temp_delet/filtered_csv.csv', img_dir_prefix=img_dir_generated, transform=transform_clip)
    generated_dataset_inception = ImageDataset(csv_file='/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/test.csv', img_dir_prefix=img_dir_generated, transform=transform_inception)
    generated_dataset_clip = ImageDataset(csv_file='/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/test.csv', img_dir_prefix=img_dir_generated, transform=transform_clip)

    if n_images:
        real_dataset_inception = torch.utils.data.Subset(real_dataset_inception, list(range(n_images)))
        real_dataset_clip = torch.utils.data.Subset(real_dataset_clip, list(range(n_images)))
        generated_dataset_inception = torch.utils.data.Subset(generated_dataset_inception, list(range(n_images)))
        generated_dataset_clip = torch.utils.data.Subset(generated_dataset_clip, list(range(n_images)))

    real_dataloader_inception = DataLoader(real_dataset_inception, batch_size=batch_size, shuffle=False)
    real_dataloader_clip = DataLoader(real_dataset_clip, batch_size=batch_size, shuffle=False)
    generated_dataloader_inception = DataLoader(generated_dataset_inception, batch_size=batch_size, shuffle=False)
    generated_dataloader_clip = DataLoader(generated_dataset_clip, batch_size=batch_size, shuffle=False)

    return real_dataloader_inception, generated_dataloader_inception, real_dataloader_clip, generated_dataloader_clip

# calculate FID for all models
def calculate_fid_for_all_models(generated_dataloader_inception, real_dataloader_inception,
                                 generated_dataloader_clip, real_dataloader_clip, 
                                 xrv_model, inception_model, clip_model, device):
    inception_features_real = get_features(inception_model, real_dataloader_inception, device, 2048)
    inception_features_generated = get_features(inception_model, generated_dataloader_inception, device, 2048)
    mu_inception_real, sigma_inception_real = compute_statistics(inception_features_real)
    mu_inception_generated, sigma_inception_generated = compute_statistics(inception_features_generated)
    fid_inception = calculate_fid(mu_inception_real, sigma_inception_real, mu_inception_generated, sigma_inception_generated)

    clip_features_real = get_features(clip_model.encode_image, real_dataloader_clip, device, 512, is_clip=True)
    clip_features_generated = get_features(clip_model.encode_image, generated_dataloader_clip, device, 512, is_clip=True)
    mu_clip_real, sigma_clip_real = compute_statistics(clip_features_real)
    mu_clip_generated, sigma_clip_generated = compute_statistics(clip_features_generated)
    fid_clip = calculate_fid(mu_clip_real, sigma_clip_real, mu_clip_generated, sigma_clip_generated)

    xrv_features_real = get_features(xrv_model, real_dataloader_inception, device, 1024)
    xrv_features_generated = get_features(xrv_model, generated_dataloader_inception, device, 1024)
    mu_xrv_real, sigma_xrv_real = compute_statistics(xrv_features_real)
    mu_xrv_generated, sigma_xrv_generated = compute_statistics(xrv_features_generated)
    fid_xrv = calculate_fid(mu_xrv_real, sigma_xrv_real, mu_xrv_generated, sigma_xrv_generated)

    return fid_inception, fid_clip, fid_xrv

# Set up device and models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
inception_model = models.inception_v3(pretrained=True, transform_input=False).to(device)
inception_model.fc = torch.nn.Identity()
xrv_model = load_xrv_model().to(device)
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# paths and checkpoint directories
csv_file = "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/test.csv"
real_image_dir = '/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/'
# these folders contain images generated by the respective generative model checkpoint from training
checkpoint_dirs = ["/workspace/my_auxiliary_persistent/retrain_roentgen/generated_test_sets/imgs/"]

#results_file_path = "/workspace/my_auxiliary_persistent/retrain_roentgen/eval_checkpoints/quick_eval_results.txt"
results_file_path = "/workspace/my_auxiliary_persistent/retrain_roentgen/eval_metrics_final/three_fids_temp.txt"

# open the results file in append mode
with open(results_file_path, "a") as results_file:
    # iterate over checkpoints, evaluate and print/save results
    for checkpoint_dir in checkpoint_dirs:
        print(f"Evaluating checkpoint: {checkpoint_dir}")
        
        # write the checkpoint name to the results file
        results_file.write(f"Evaluating checkpoint: {checkpoint_dir}\n")

        # create dataloaders
        real_dataloader_inception, generated_dataloader_inception, real_dataloader_clip, generated_dataloader_clip = create_dataloaders(
            csv_file, real_image_dir, checkpoint_dir, n_images=100)#n_images=10000)

        # FID scores
        fid_inception, fid_clip, fid_xrv = calculate_fid_for_all_models(
            generated_dataloader_inception, real_dataloader_inception,
            generated_dataloader_clip, real_dataloader_clip,
            xrv_model, inception_model, clip_model, device)

        print(f"Results for {checkpoint_dir}:")
        print(f"FID InceptionV3: {fid_inception}")
        print(f"FID CLIP-ViT-B/32: {fid_clip}")
        print(f"FID DenseNet-121 (XRV): {fid_xrv}")
        print("\n" + "="*50 + "\n")


        results_file.write(f"Results for {checkpoint_dir}:\n")
        results_file.write(f"FID InceptionV3: {fid_inception}\n")
        results_file.write(f"FID CLIP-ViT-B/32: {fid_clip}\n")
        results_file.write(f"FID DenseNet-121 (XRV): {fid_xrv}\n")
        results_file.write("\n" + "="*50 + "\n\n")

