import pandas as pd

# paths to remove
# OLD
#paths_to_remove = [
#    "/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0_batch_2_train_1/patient09797/study4/view1_frontal.jpg",
#    "/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0_batch_2_train_1/patient05271/study5/view2_frontal.jpg",
#    "/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0_batch_2_train_1/patient21394/study1/view1_frontal.jpg"
#]

# NEW retrain roentgen on test set
paths_to_remove = [
    "/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0_batch_2_train_1/patient20683/study2/view1_frontal.jpg",
    "/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0_batch_3_train_2/patient33492/study1/view2_frontal.jpg",
    "/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0_batch_2_train_1/patient10603/study2/view2_frontal.jpg",
]


# OLD
#csv_path = "/workspace/persistent/code/roentgen/weights_and_bobs/diffusion_splits/processed_train_with_projection_corr_rot.csv"
# NEW retrain roentgen on test set
csv_path = "/workspace/my_auxiliary_persistent/retrain_roentgen/csvs/diffusion_splits/processed_train_with_projection_corr_rot.csv"
df = pd.read_csv(csv_path)

# print the number of rows before removal
num_rows_before = df.shape[0]
print(f"Number of rows before removal: {num_rows_before}")

# remove the specified rows
df_filtered = df[~df['file_name'].isin(paths_to_remove)]

# prinye number of rows after removal
num_rows_after = df_filtered.shape[0]
print(f"Number of rows after removal: {num_rows_after}")

# save the updated CSV file with a new name
new_csv_path = csv_path.replace(".csv", "_removed_bad.csv")
df_filtered.to_csv(new_csv_path, index=False)

print(f"Updated CSV saved to {new_csv_path}")

