import os
import pandas as pd
import matplotlib.pyplot as plt

# test dirs containing experiment/trial info
test_dir = "/workspace/my_auxiliary_persistent/natmed_multitrial_results_synthetic_data/"
fig_out_dir = "/workspace/my_auxiliary_persistent/natmed_multitrial_results_plots/synthetic_data_results.png"


def extract_auc_ci_and_samples(test_dir):
    folder_names = []
    auc_means = []
    auc_cis = []
    sample_counts = []
    

    for folder_name in os.listdir(test_dir):
        folder_path = os.path.join(test_dir, folder_name)
        
       
        if os.path.isdir(folder_path):
            summary_csv_path = os.path.join(folder_path, f"{folder_name}_summary.csv")
            true_trial_csv_path = os.path.join(folder_path, "True_trial_0.csv")
            
            if os.path.exists(summary_csv_path):
               
                summary_df = pd.read_csv(summary_csv_path)
                
               
                auc_row = summary_df[summary_df['Metric'] == 'Mean AUC']
                if not auc_row.empty:
                    mean_auc = auc_row['Mean'].values[0]
                    auc_ci = auc_row['Confidence Interval'].values[0]
                    
                
                    folder_names.append(folder_name)
                    auc_means.append(mean_auc)
                    auc_cis.append(auc_ci)
                    
                
                    if os.path.exists(true_trial_csv_path):
                        true_trial_df = pd.read_csv(true_trial_csv_path)
                        sample_count = len(true_trial_df)
                    else:
                        sample_count = 0  # Set to 0 if the file is missing
                    sample_counts.append(sample_count)
    
    return folder_names, auc_means, auc_cis, sample_counts


folder_names, auc_means, auc_cis, sample_counts = extract_auc_ci_and_samples(test_dir)

# sort the folder names alphabetically and rearrange the acu means, CIs and sample counts accordingly
sorted_data = sorted(zip(folder_names, auc_means, auc_cis, sample_counts))
folder_names, auc_means, auc_cis, sample_counts = zip(*sorted_data)

#  labels with sample count for the x-axis
folder_labels_with_samples = [f"{folder_name} (n={sample_count})" for folder_name, sample_count in zip(folder_names, sample_counts)]

# Plotting the mean auC values with confidence intervals as a bar chart
plt.figure(figsize=(10, 6))
x_positions = range(len(folder_names))


plt.bar(x_positions, auc_means, color='skyblue', yerr=auc_cis, capsize=5, ecolor='red', edgecolor='blue')


plt.xlabel('Experiment Folders')
plt.ylabel('Mean AUC')
plt.title('Mean AUC Across Experiments with Confidence Intervals')


plt.xticks(x_positions, folder_labels_with_samples, rotation=45, ha='right')  # Rotate folder names on x-axis for better readability
plt.tight_layout()


plt.ylim(min(auc_means) - max(auc_cis), max(auc_means) + max(auc_cis))


plt.savefig(fig_out_dir)


plt.clf()
