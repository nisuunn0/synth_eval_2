import os
import pandas as pd
import matplotlib.pyplot as plt
import re


base_test_dirs = [
    "results_ground_truth", 
    "results_test_fully_synthetic", 
    "results_test_real", 
    "results_test_upsampled"
]


#base_path = "/workspace/my_auxiliary_persistent/natmed_multitrial_results_synthetic_data/"
base_path = "/workspace/my_auxiliary_persistent/upsampled_test_set/"


fig_out_dir = "/workspace/my_auxiliary_persistent/upsampled_test_set/plots/synthetic_data_results.png"
############################################################################



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
                    # Read the summary CSV
                    summary_df = pd.read_csv(summary_csv_path)

                    # get mean auc and its confidence interval
                    auc_row = summary_df[summary_df['Metric'] == 'Mean AUC']
                    if not auc_row.empty:
                        mean_auc = auc_row['Mean'].values[0]
                        auc_ci = auc_row['Confidence Interval'].values[0]

                        # get the number of samples from the "True_trial_0.csv" file
                        if os.path.exists(true_trial_csv_path):
                            true_trial_df = pd.read_csv(true_trial_csv_path)
                            sample_count = len(true_trial_df)
                        else:
                            sample_count = 0  # set to 0 if the file is missing

                      
                        match = re.search(r'(\d+-\d+|80\+)_?(Male|Female)?', folder_name)
                        age_group = match.group(1) if match else "Unknown"
                        sex_group = match.group(2) if match and match.group(2) else "Unknown"
                        group_label = f"{age_group}_{sex_group}"

                        # Store folder_name (removing "results_" and everything after the first slash)
                        clean_folder_name = base_dir.replace("results_", "")
                        data.append((clean_folder_name, mean_auc, auc_ci, sample_count, group_label))

    return data


data = extract_auc_ci_and_samples(base_test_dirs, base_path)


order_of_tests = ["ground_truth", "test_real", "test_upsampled", "test_fully_synthetic"]
data.sort(key=lambda x: (x[4], order_of_tests.index(x[0])))


plt.figure(figsize=(14, 8))

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
          
            current_position += 1
        group_start_positions.append(current_position)
        group_labels.append(group_label)
        group_to_bars[group_label] = []  
        previous_group_label = group_label


    x_label = f"{clean_folder_name} (n={sample_count})"  
    # Plot the bar
    plt.bar(current_position, mean_auc, color='skyblue', yerr=auc_ci, capsize=5, ecolor='red', edgecolor='blue')

  
    group_to_bars[group_label].append(current_position)
    x_positions.append(current_position) 
    current_position += 1


plt.xticks(x_positions, [f"{entry[0]} (n={entry[3]})" for entry in data], rotation=45, ha='right')


for group_label, bar_positions in group_to_bars.items():
    group_center = (bar_positions[0] + bar_positions[-1]) / 2  # Center of the group
    plt.text(group_center, max(mean_aucs) + max(auc_cis) * 0.2, group_label, ha='center', fontsize=10, rotation=45)


plt.xlabel('Experiment Folders')
plt.ylabel('Mean AUC')
#plt.title('Mean AUC Across Experiments Grouped by Age and Sex')


plt.ylim(min(mean_aucs) - max(auc_cis), max(mean_aucs) + max(auc_cis))

plt.tight_layout()


plt.savefig(fig_out_dir)


plt.clf()

