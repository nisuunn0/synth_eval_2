import pandas as pd
import os
from itertools import combinations


input_csv = "/workspace/my_auxiliary_persistent/retrain_roentgen/consistency_test_subpop_level_summary/consistency_subpop_level_summary.csv"


output_csv = "/workspace/my_auxiliary_persistent/retrain_roentgen/consistency_test_subpop_level_summary/final_subpop_consistency_values.csv"


def compute_consistency(row):
    real_values = []
    synthetic_values = []

    # get AUC values for real and synthetic columns
    for model in ["DenseNet121", "Inception v3", "ResNet-50", "MobileNet V3Small"]:
        real_col = f"{model} Real"
        synth_col = f"{model} Synthetic"
        if real_col in row and synth_col in row:
            real_values.append(float(row[real_col].split("±")[0]))
            synthetic_values.append(float(row[synth_col].split("±")[0]))
    
    # calce consistency
    consistencies_count = 0
    total_pairs = len(real_values) * (len(real_values) - 1) / 2.0  # total pairs of modelss
    for i, j in combinations(range(len(real_values)), 2):
        real_diff = real_values[i] - real_values[j]
        synth_diff = synthetic_values[i] - synthetic_values[j]
        if (real_diff > 0 and synth_diff > 0) or (real_diff < 0 and synth_diff < 0) or (real_diff == 0 and synth_diff == 0):
            consistencies_count += 1

    return consistencies_count / total_pairs if total_pairs > 0 else 0


df = pd.read_csv(input_csv)

consistency for each row
consistency_results = []
for _, row in df.iterrows():
    consistency = compute_consistency(row)
    consistency_results.append({"Age Group": row["Age Group"], "Consistency": round(consistency, 4)})


consistency_df = pd.DataFrame(consistency_results)


os.makedirs(os.path.dirname(output_csv), exist_ok=True)
consistency_df.to_csv(output_csv, index=False)

print(f"Consistency metrics have been computed and saved to {output_csv}.")
