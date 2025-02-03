import os
import pandas as pd


csv_file = "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/test.csv"
image_prefix = "/workspace/my_auxiliary_persistent/retrain_roentgen/test_checkpoints_images/checkpoint-12500/"
output_csv_file = "/workspace/my_auxiliary_persistent/retrain_roentgen/eval_checkpoints/filtered_test.csv"


df = pd.read_csv(csv_file)


def image_exists(relative_path):
    full_path = os.path.join(image_prefix, relative_path)
    return os.path.exists(full_path)


filtered_df = df[df['Path'].apply(image_exists)]


filtered_df.to_csv(output_csv_file, index=False)

print(f"filtered CSV saved successfully at {output_csv_file}")
