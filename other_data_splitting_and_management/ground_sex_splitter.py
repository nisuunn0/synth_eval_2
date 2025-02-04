import pandas as pd
import os

# path to the directory containing the age group  files
input_dir = 'my_splits_further_test'

# ist all the csvs in directory
csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]


output_dir = 'my_splits_further_sex_test'
os.makedirs(output_dir, exist_ok=True)


for csv_file in csv_files:
  
    file_path = os.path.join(input_dir, csv_file)
    df = pd.read_csv(file_path)
    
    
    sexes = df['Sex'].unique()
    
   
    for sex in sexes:
        sex_df = df[df['Sex'] == sex]
        sex_file_path = os.path.join(output_dir, f'{csv_file[:-4]}_{sex}.csv')
        sex_df.to_csv(sex_file_path, index=False)

print("csv files have been further split by sex and saved.")

