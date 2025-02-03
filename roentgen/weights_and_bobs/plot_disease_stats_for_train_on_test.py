import os
import pandas as pd
import matplotlib.pyplot as plt


root_dir = '/workspace/my_auxiliary_persistent/retrain_roentgen/train_on_test/results'


output_dir = '/workspace/my_auxiliary_persistent/retrain_roentgen/train_on_test/disease_results_plots'

os.makedirs(output_dir, exist_ok=True)

for experiment_dir in os.listdir(root_dir):
    experiment_path = os.path.join(root_dir, experiment_dir)
    #  only processing directories
    if os.path.isdir(experiment_path):
        # path to the specific CSV file
        csv_path = os.path.join(experiment_path, 'multi_trial_results/valid/valid_disease_summary.csv')
        if os.path.exists(csv_path):
            try:
                # read the csv file, using multiple delimiters to account for potential formatting issues
                data = pd.read_csv(csv_path, delimiter=None, engine='python')
                
                # clean column names to ensure no formatting issues
                data.columns = data.columns.str.strip()

                # rxtract Disease and Mean AUC columns
                if 'Disease' in data.columns and 'Mean AUC' in data.columns:
                    diseases = data['Disease']
                    mean_auc = data['Mean AUC']
                    
                    # plotting
                    plt.figure(figsize=(10, 6))
                    plt.bar(diseases, mean_auc, color='skyblue')
                    plt.xlabel('Disease', fontsize=12)
                    plt.ylabel('Mean AUC', fontsize=12)
                    plt.title(f'{experiment_dir} - Mean AUC per Disease', fontsize=14)
                    plt.xticks(rotation=45, ha='right', fontsize=10)
                    
                    plt.ylim(0.0, 1.0)
                    
                    plt.tight_layout()
                        
                    # save the plot
                    output_file = os.path.join(output_dir, f'{experiment_dir}_auc_plot.png')
                    plt.savefig(output_file)
                    plt.close()
                    
                    print(f"Plot saved for {experiment_dir}: {output_file}")
                else:
                    print(f"Missing required columns in {csv_path}")
            except Exception as e:
                print(f"Error processing {csv_path}: {e}")

