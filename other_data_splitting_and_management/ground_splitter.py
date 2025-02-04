import pandas as pd
import os

source_csv_path = 'my_splits/test.csv'


df = pd.read_csv(source_csv_path)


age_groups = df['Age Group'].unique()


output_dir = 'my_splits_further_test'
os.makedirs(output_dir, exist_ok=True)


for age_group in age_groups:
    group_df = df[df['Age Group'] == age_group]
    output_file_path = os.path.join(output_dir, f'ground_truth_{age_group}.csv')
    group_df.to_csv(output_file_path, index=False)

print("csv files have been split by age group and saved.")

