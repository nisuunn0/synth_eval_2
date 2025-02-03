import os
import pandas as pd
import matplotlib.pyplot as plt
import re


base_test_dirs_old = [
    "results_ground_truth",
    "results_test_real"
]
base_test_dirs_new = [
    "results_test_fully_synthetic",
    "results_test_upsampled"
]


base_path_old = "/workspace/my_auxiliary_persistent/upsampled_test_set/"
base_path_new = "/workspace/my_auxiliary_persistent/retrain_roentgen/generated_test_sets/results/"


#fig_out_dir = "/workspace/my_auxiliary_persistent/retrain_roentgen/generated_test_sets/plots_of_results/multi_trial_stats_try_6.png" # multi_trial_stats_try_6.png is latest most correct image
fig_out_dir = "/workspace/my_auxiliary_persistent/retrain_roentgen/generated_test_sets/plots_of_results/multi_trial_stats_try_7.png" # try without majority age group (40-80)


additional_groups = [
    # Group 1: Female
    {
        "test_real": {"label": "test_real (n=1000)", "AUC": 0.814, "CI": 0.000834},
        "test_synthetic": {"label": "test_synthetic (n=2000)", "AUC": 0.839, "CI": 0.00065},
        "age_sex": "40_80_Female"
    },
    # Group 2: Male
    {
        "test_real": {"label": "test_real (n=3000)", "AUC": 0.816, "CI": 0.0006},
        "test_synthetic": {"label": "test_synthetic (n=4000)", "AUC": 0.837, "CI": 0.0005},
        "age_sex": "40_80_Male"
    },
    # Group 3: General 40-80 group
    {
        "test_real": {"label": "test_real (n=5000)", "AUC": 0.816, "CI": 0.0005},
        "test_synthetic": {"label": "test_synthetic (n=6000)", "AUC": 0.838, "CI": 0.0004},
        "age_sex": "40_80"
    }
]


def extract_auc_ci_and_samples(base_test_dirs, base_path):
    data = []
    for base_dir in base_test_dirs:
        full_base_dir = os.path.join(base_path, base_dir)
        if not os.path.exists(full_base_dir):
            continue
        for folder_name in os.listdir(full_base_dir):
            folder_path = os.path.join(full_base_dir, folder_name)
            if os.path.isdir(folder_path):
                summary_csv_path = os.path.join(folder_path, f"{folder_name}_summary.csv")
                true_trial_csv_path = os.path.join(folder_path, "True_trial_0.csv")
                if os.path.exists(summary_csv_path):
                    summary_df = pd.read_csv(summary_csv_path)
                    auc_row = summary_df[summary_df['Metric'] == 'Mean AUC']
                    if not auc_row.empty:
                        mean_auc = auc_row['Mean'].values[0]
                        auc_ci = auc_row['Confidence Interval'].values[0]
                        sample_count = len(pd.read_csv(true_trial_csv_path)) if os.path.exists(true_trial_csv_path) else 0
                        match = re.search(r'(\d+-\d+|80\+)_?(Male|Female)?', folder_name)
                        age_group = match.group(1) if match else "Unknown"
                        sex_group = match.group(2) if match and match.group(2) else "Unknown"
                        group_label = f"{age_group}_{sex_group}"
                        clean_folder_name = base_dir.replace("results_", "")
                        data.append((clean_folder_name, mean_auc, auc_ci, sample_count, group_label))
    return data


data_old = extract_auc_ci_and_samples(base_test_dirs_old, base_path_old)
data_new = extract_auc_ci_and_samples(base_test_dirs_new, base_path_new)


data = data_old + data_new
order_of_tests = ["ground_truth", "test_real", "test_upsampled", "test_fully_synthetic"]
data.sort(key=lambda x: (x[4], order_of_tests.index(x[0])))


plt.figure(figsize=(16, 8))  # Increase figure width for more spacing
x_positions = []
current_position = 0
group_start_positions = []
group_labels = []
group_to_bars = {}
mean_aucs = [entry[1] for entry in data]
auc_cis = [entry[2] for entry in data]
previous_group_label = None


for i, (clean_folder_name, mean_auc, auc_ci, sample_count, group_label) in enumerate(data):
    if group_label != previous_group_label:
        if previous_group_label is not None:
            current_position += 2  # Increase space between groups
        group_start_positions.append(current_position)
        group_labels.append(group_label)
        group_to_bars[group_label] = []
        previous_group_label = group_label

    plt.bar(current_position, mean_auc, color='skyblue', yerr=auc_ci, capsize=5, ecolor='red', edgecolor='blue')
    group_to_bars[group_label].append(current_position)
    x_positions.append(current_position)
    current_position += 1


current_position += 5 
    

gap_between_additional_groups = 3  
gap_between_bars_within_group = 0.7  
additional_group_labels = []  

'''
for group in additional_groups:

    real_data = group["test_real"]
    plt.bar(current_position, real_data["AUC"], color="lightgreen", yerr=real_data["CI"], capsize=5, label=real_data["label"], edgecolor="green")
    x_positions.append(current_position)
    additional_group_labels.append(real_data["label"])  # Label for the 'test_real' bar
    current_position += gap_between_bars_within_group  # Adjust spacing for bars within a group

 
    synthetic_data = group["test_synthetic"]
    plt.bar(current_position, synthetic_data["AUC"], color="orange", yerr=synthetic_data["CI"], capsize=5, label=synthetic_data["label"], edgecolor="darkorange")
    x_positions.append(current_position)
    additional_group_labels.append(synthetic_data["label"])  # Label for the 'test_synthetic' bar
    current_position += 1  # Move to the next group

    # Add age and sex information above the bars
    group_center = (x_positions[-2] + x_positions[-1]) / 2
    plt.text(group_center, max(mean_aucs) + max(auc_cis) * 0.2, group["age_sex"], ha="center", fontsize=12, rotation=45)

    # increase the gap between groups
    current_position += gap_between_additional_groups
'''


all_labels = [f"{entry[0]} (n={entry[3]})" for entry in data] #+ additional_group_labels # add this if plotting 40-80 majority group as well
plt.xticks(x_positions, all_labels, rotation=45, ha='right')


for group_label, bar_positions in group_to_bars.items():
    group_center = (bar_positions[0] + bar_positions[-1]) / 2
    plt.text(group_center, max(mean_aucs) + max(auc_cis) * 0.2, group_label, ha='center', fontsize=10, rotation=45)


plt.xlabel('Experiment Folders')
plt.ylabel('Mean AUC')
plt.ylim(min(mean_aucs) - max(auc_cis), max(mean_aucs) + max(auc_cis))
plt.tight_layout()
plt.savefig(fig_out_dir)
plt.clf()
