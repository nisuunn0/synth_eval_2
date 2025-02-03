import os
import pandas as pd


root_dir = '/workspace/my_auxiliary_persistent/retrain_roentgen/train_on_test/results'


output_dir = '/workspace/my_auxiliary_persistent/retrain_roentgen/train_on_test/disease_results_plots'
os.makedirs(output_dir, exist_ok=True)

summary_csv_path = os.path.join(output_dir, 'disease_summary_results_all_together.csv')


summary_data = []

# traverse through all folders in the results directory
for experiment_dir in os.listdir(root_dir):
    experiment_path = os.path.join(root_dir, experiment_dir)
    if os.path.isdir(experiment_path):
        
        csv_path = os.path.join(experiment_path, 'multi_trial_results/valid/valid_disease_summary.csv')
        if os.path.exists(csv_path):
            try:
                
                data = pd.read_csv(csv_path, delimiter=None, engine='python')
                
                # clean column names 
                data.columns = data.columns.str.strip()

                
                if 'Disease' in data.columns and 'Mean AUC' in data.columns:
                    diseases = data['Disease']
                    mean_auc = data['Mean AUC']
                    
                
                    result_row = {'Experiment': experiment_dir}
                    for disease, auc in zip(diseases, mean_auc):
                        #result_row[disease] = auc # ORIGINAL
                        result_row[disease] = round(auc, 3) # round to 3 dp
                    summary_data.append(result_row)
                else:
                    print(f"Missing required columns in {csv_path}")
            except Exception as e:
                print(f"Error processing {csv_path}: {e}")


if summary_data:
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Aggregated results saved to {summary_csv_path}")
else:
    print("No data available to save in the summary CSV.")



