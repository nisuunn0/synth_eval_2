import os
import pandas as pd
import numpy as np

# compute metrics from confusion matrix components
def compute_metrics(TP, TN, FP, FN):
    metrics = {}
    

    metrics["accuracy"] = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
    metrics["precision"] = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    metrics["recall"] = TP / (TP + FN) if (TP + FN) > 0 else 0.0  # Also known as sensitivity or TPR
    metrics["specificity"] = TN / (TN + FP) if (TN + FP) > 0 else 0.0  # TNR
    metrics["f1_score"] = 2 * (metrics["precision"] * metrics["recall"]) / (metrics["precision"] + metrics["recall"]) if (metrics["precision"] + metrics["recall"]) > 0 else 0.0
    metrics["FPR"] = FP / (FP + TN) if (FP + TN) > 0 else 0.0  # False positive rate
    metrics["FNR"] = FN / (FN + TP) if (FN + TP) > 0 else 0.0  # False negative rate
    metrics["prevalence"] = (TP + FN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
    metrics["P"] = TP + FN  # Positive cases
    metrics["N"] = TN + FP  # Negative cases
    
    return metrics

# process folders and compute metrics
def process_folders(input_folders, output_base):
    class_labels = [
        "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema", 
        "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other", 
        "Fracture", "Support Devices"
    ]
    
    for folder in input_folders:
        folder_name = os.path.basename(os.path.normpath(folder))  
        for subfolder in os.listdir(folder):
            subfolder_path = os.path.join(folder, subfolder)
            if not os.path.isdir(subfolder_path):
                continue

            # output directory structure including the general folder name
            output_dir = os.path.join(output_base, folder_name, subfolder)
            os.makedirs(output_dir, exist_ok=True)

            trials_metrics = {label: [] for label in class_labels}
            overall_metrics_per_trial = []  # List to store overall metrics per trial

            # loop through trials (0-49)
            for trial in range(50):
                true_labels_path = os.path.join(subfolder_path, f"True_trial_{trial}.csv")
                pred_labels_path = os.path.join(subfolder_path, f"bipred_trial_{trial}.csv")

                if not os.path.exists(true_labels_path) or not os.path.exists(pred_labels_path):
                    continue

                true_labels_df = pd.read_csv(true_labels_path)
                pred_labels_df = pd.read_csv(pred_labels_path)

                true_labels = true_labels_df.drop(columns=["Path"]).values
                pred_labels = pred_labels_df.drop(columns=["Path"])

                # Ensure that we are only converting boolean True/False to 1.0/0.0
                pred_labels.replace({True: 1.0, False: 0.0}, inplace=True)

                # init aggregated values for the overall confusion matrix
                TP_total, TN_total, FP_total, FN_total = 0, 0, 0, 0

                # compute confusion matrix components and metrics for each label
                for i, label in enumerate(class_labels):
                    TP = np.sum((true_labels[:, i] == 1) & (pred_labels.iloc[:, i] == 1))
                    TN = np.sum((true_labels[:, i] == 0) & (pred_labels.iloc[:, i] == 0))
                    FP = np.sum((true_labels[:, i] == 0) & (pred_labels.iloc[:, i] == 1))
                    FN = np.sum((true_labels[:, i] == 1) & (pred_labels.iloc[:, i] == 0))

                    # compute metrics for the current class and store them
                    metrics = compute_metrics(TP, TN, FP, FN)
                    metrics["TP"] = TP
                    metrics["TN"] = TN
                    metrics["FP"] = FP
                    metrics["FN"] = FN
                    trials_metrics[label].append(metrics)

                    # aggregate the values for overall metrics
                    TP_total += TP
                    TN_total += TN
                    FP_total += FP
                    FN_total += FN

                # compute overall metrics for this trial
                overall_metrics = compute_metrics(TP_total, TN_total, FP_total, FN_total)
                overall_metrics_per_trial.append(overall_metrics)

            # save per-trial metrics for each class
            for label in class_labels:
                per_trial_metrics_path = os.path.join(output_dir, f"metrics_per_trial_{label}.csv")
                pd.DataFrame(trials_metrics[label]).to_csv(per_trial_metrics_path, index=False)

            # Save average metrics for each class
            for label in class_labels:
                avg_metrics = pd.DataFrame(trials_metrics[label]).mean(axis=0).to_dict()
                avg_metrics_path = os.path.join(output_dir, f"avg_metrics_{label}.csv")
                pd.DataFrame([avg_metrics]).to_csv(avg_metrics_path, index=False)

            # save overall metrics for each trial (across all classes)
            overall_metrics_per_trial_path = os.path.join(output_dir, "overall_metrics_per_trial.csv")
            pd.DataFrame(overall_metrics_per_trial).to_csv(overall_metrics_per_trial_path, index=False)

            #save average overall metrics (across all trials)
            avg_overall_metrics = pd.DataFrame(overall_metrics_per_trial).mean(axis=0).to_dict()
            avg_overall_metrics_path = os.path.join(output_dir, "avg_overall_metrics.csv")
            pd.DataFrame([avg_overall_metrics]).to_csv(avg_overall_metrics_path, index=False)

if __name__ == "__main__":
    input_folders = [
        "/workspace/my_auxiliary_persistent/retrain_roentgen/generated_test_sets/results/results_test_fully_synthetic/",
        "/workspace/my_auxiliary_persistent/retrain_roentgen/generated_test_sets/results/results_test_upsampled/",
        "/workspace/my_auxiliary_persistent/upsampled_test_set/results_ground_truth/",
        "/workspace/my_auxiliary_persistent/upsampled_test_set/results_test_real/",
        "/workspace/my_auxiliary_persistent/retrain_roentgen/make_sure_i_have_them_results/test_on_real_test_set_40-80/",
        "/workspace/my_auxiliary_persistent/retrain_roentgen/make_sure_i_have_them_results/test_on_synthetic_test_set_40-80/",
    ]
    
    output_base = "/workspace/my_auxiliary_persistent/retrain_roentgen/generated_test_sets/confusion_matrices_accompanying_stats_not_disease_specific/"
    process_folders(input_folders, output_base)
