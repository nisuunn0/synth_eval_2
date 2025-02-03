import pandas as pd

#  update image paths to absolute paths
def update_image_paths(csv_file, prefix):
    df = pd.read_csv(csv_file)
    df['file_name'] = prefix + df['file_name']  # update img paths
    return df

# txt prompt based on csv 
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
        if row[condition] == 1:
            prompt += f", {condition}"
    #print(prompt)
    return prompt.strip(", ")

# 
def process_csv(csv_file, prefix):
    df = pd.read_csv(csv_file)
    df['Path'] = prefix + df['Path']  # Update image paths
    df['text'] = df.apply(generate_text_prompt, axis=1)  # Generate text prompt
    return df[['Path', 'text']]  # Select only necessary columns


#train_csv_file = "/workspace/persistent/code/roentgen/weights_and_bobs/diffusion_splits/train_from_valid.csv" # OLD 
#valid_csv_file = "/workspace/persistent/code/roentgen/weights_and_bobs/diffusion_splits/valid_from_valid.csv" # OLD
train_csv_file = "/workspace/my_auxiliary_persistent/retrain_roentgen/csvs/diffusion_splits/train_from_test.csv" # NEW retrain roentgen on test set
valid_csv_file = "/workspace/my_auxiliary_persistent/retrain_roentgen/csvs/diffusion_splits/valid_from_test.csv" # NEW retrain roentgen on test set
prefix = "/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/"


train_df = process_csv(train_csv_file, prefix)
train_df.columns = ['file_name', 'text']  # Rename columns


valid_df = process_csv(valid_csv_file, prefix)
valid_df.columns = ['file_name', 'text']  # Rename columns


#train_df.to_csv("/workspace/persistent/code/roentgen/weights_and_bobs/diffusion_splits/processed_train_with_projection.csv", index=False) # OLD
#valid_df.to_csv("/workspace/persistent/code/roentgen/weights_and_bobs/diffusion_splits/processed_valid_with_projection.csv", index=False) # OLD
train_df.to_csv("/workspace/my_auxiliary_persistent/retrain_roentgen/csvs/diffusion_splits/processed_train_with_projection.csv", index=False) # NEW retrain roentgen on test set
valid_df.to_csv("/workspace/my_auxiliary_persistent/retrain_roentgen/csvs/diffusion_splits/processed_valid_with_projection.csv", index=False) # NEW retrain roentgen on test set


