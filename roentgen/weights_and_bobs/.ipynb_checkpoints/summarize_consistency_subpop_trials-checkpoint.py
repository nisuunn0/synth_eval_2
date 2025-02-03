import os
import pandas as pd
import numpy as np

#folders with experiments
base_dir = "/workspace/my_auxiliary_persistent/retrain_roentgen/consistency_test_subpop_level/"


models = ["densenet", "inception", "resnet", "mobilenet"]
data_types = ["real", "synth"]

# get mean auc and CI
def parse_summary_csv(file_path):
    df = pd.read_csv(file_path)
    mean_auc = df.loc[df["Metric"] == "Mean AUC", "Mean"].values[0]
    ci = df.loc[df["Metric"] == "Mean AUC", "Confidence Interval"].values[0]
    return mean_auc, ci

# map folder names to Age Group entries from the given template
def extract_age_group(folder_name):
    age_group = folder_name.replace("test_", "").replace("_", " ").capitalize()
    return age_group

# Initt. an emptydf to store results
columns = ["Age Group", "DenseNet121 Real", "DenseNet121 Synthetic", "Inception v3 Real", "Inception v3 Synthetic", 
           "ResNet-50 Real", "ResNet-50 Synthetic", "MobileNet V3Small Real", "MobileNet V3Small Synthetic"]
result_df = pd.DataFrame(columns=columns)

# Expected order of rows based on the provided template
expected_order = ["0-20", "0-20 female", "0-20 male", "20-40", "20-40 female", "20-40 male", 
                  "40-80", "40-80 female", "40-80 male", "80+", "80+ female", "80+ male"]

# iterate over each model and data type
for model in models:
    for data_type in data_types:
        # determine the appropriate column name for the csv output
        model_name = "DenseNet121" if model == "densenet" else \
                     "Inception v3" if model == "inception" else \
                     "ResNet-50" if model == "resnet" else \
                     "MobileNet V3Small"
        column_name = f"{model_name} {'Real' if data_type == 'real' else 'Synthetic'}"

        # get the directory for the current model and data type
        model_dir = os.path.join(base_dir, f"{model}_{data_type}")

        # iterate over all subdirectories (age groups)
        for age_group_folder in os.listdir(model_dir):
            age_group_path = os.path.join(model_dir, age_group_folder)
            if os.path.isdir(age_group_path):
                summary_file = os.path.join(age_group_path, f"{age_group_folder}_summary.csv")
                if os.path.exists(summary_file):
                   
                    age_group = extract_age_group(age_group_folder)

                    
                    mean_auc, ci = parse_summary_csv(summary_file)

                    
                    formatted_value = f"{mean_auc:.3f}±{ci:.3f}"

                    
                    if age_group not in result_df["Age Group"].values:
                        result_df = pd.concat([result_df, pd.DataFrame({"Age Group": [age_group]})], ignore_index=True)

                    result_df.loc[result_df["Age Group"] == age_group, column_name] = formatted_value

# Reorder rows to match the expected order
result_df["Age Group"] = pd.Categorical(result_df["Age Group"], categories=expected_order, ordered=True)
result_df = result_df.sort_values("Age Group").reset_index(drop=True)


output_csv_path = "/workspace/my_auxiliary_persistent/retrain_roentgen/consistency_test_subpop_level_summary/consistency_subpop_level_summary.csv"
result_df.to_csv(output_csv_path, index=False)

print(f"CSV table has been filled and saved to {output_csv_path}.")
