import pandas as pd





PATH_TO_IMAGES_REAL = "/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/"

PATH_TO_IMAGES_SYNTHETIC = "/workspace/my_auxiliary_persistent/retrain_roentgen/full_test_set_for_metric_assessment/checkpoint-44000/"



# original training CSV file

TRAIN_DF_PATH = "/workspace/my_auxiliary_persistent/retrain_roentgen/train_on_test/test_split_to_train_and_valid/train_split.csv"

df = pd.read_csv(TRAIN_DF_PATH)



# add the real world path prefix to the 'Path' column for the original dataset
df['Path'] = PATH_TO_IMAGES_REAL + df['Path']



# reload the original CSV file (to maintain relative paths for synthetic data paths)
df_original = pd.read_csv(TRAIN_DF_PATH)



# add the synthetic path prefix to the 'Path' column for the newly appended data
df_original['Path'] = PATH_TO_IMAGES_SYNTHETIC + df_original['Path']



# append the real dataset with updated paths and the synthetic dataset
df_combined = pd.concat([df, df_original], ignore_index=True)



# save the final combined CSV to the specified directory

output_csv_path = "/workspace/my_auxiliary_persistent/retrain_roentgen/train_on_test/real_and_synth_data_together_csvs/full_real_plus_full_synth.csv"
df_combined.to_csv(output_csv_path, index=False)



print(f"Combined CSV file saved to: {output_csv_path}")

