import os
import pandas as pd
import matplotlib.pyplot as plt


test_dir = "/workspace/my_auxiliary_persistent/natmed_multitrial_results_synthetic_data/"

fig_out_dir = "/workspace/my_auxiliary_persistent/natmed_multitrial_results_plots_synth_ground"


def extract_auc_and_ci(test_dir):
    folder_names = []
    auc_means = []
    auc_cis = []
    
 
    for folder_name in os.listdir(test_dir):
        folder_path = os.path.join(test_dir, folder_name)
        
    
        if os.path.isdir(folder_path):
            summary_csv_path = os.path.join(folder_path, f"{folder_name}_summary.csv")
            
            if os.path.exists(summary_csv_path):
               
                summary_df = pd.read_csv(summary_csv_path)
                
             
                auc_row = summary_df[summary_df['Metric'] == 'Mean AUC']
                if not auc_row.empty:
                    mean_auc = auc_row['Mean'].values[0]
                    auc_ci = auc_row['Confidence Interval'].values[0]
                    
            
                    folder_names.append(folder_name)
                    auc_means.append(mean_auc)
                    auc_cis.append(auc_ci)
    
    return folder_names, auc_means, auc_cis


folder_names, auc_means, auc_cis = extract_auc_and_ci(test_dir)


plt.figure(figsize=(10, 6))
plt.errorbar(folder_names, auc_means, yerr=auc_cis, fmt='o', capsize=5, color='blue', ecolor='lightgray', elinewidth=2, markeredgewidth=2)


plt.xlabel('Experiment Folders')
plt.ylabel('Mean AUC')
plt.title('Mean AUC Across Experiments with Confidence Intervals')
plt.xticks(rotation=45, ha='right')  # Rotate folder names on x-axis for better readability
plt.tight_layout()

# save the plot
plt.savefig(fig_out_dir)

# optionally, clear the plot after saving to free up memory
plt.clf()


