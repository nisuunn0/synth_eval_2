import os
import pandas as pd
import numpy as np


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

           
            output_dir = os.path.join(output_base, folder_name, subfolder)
            os.makedirs(output_dir, exist_ok=True)

            trials_metrics = {label: [] for label in class_labels}

            # trials (0-49)
            for trial in range(50):
                true_labels_path = os.path.join(subfolder_path, f"True_trial_{trial}.csv")
                pred_labels_path = os.path.join(subfolder_path, f"bipred_trial_{trial}.csv")

                if not os.path.exists(true_labels_path) or not os.path.exists(pred_labels_path):
                    continue

                true_labels_df = pd.read_csv(true_labels_path)
                pred_labels_df = pd.read_csv(pred_labels_path)

                # drop 'Path' column from both true and predicted labels
                true_labels = true_labels_df.drop(columns=["Path"]).values
                pred_labels = pred_labels_df.drop(columns=["Path"])

                # ensure that we are only converting boolean True/False to 1.0/0.0
                pred_labels.replace({True: 1.0, False: 0.0}, inplace=True)

                # compute confusion matrix components and metrics for each label
                for i, label in enumerate(class_labels):
                    TP = np.sum((true_labels[:, i] == 1) & (pred_labels.iloc[:, i] == 1))
                    TN = np.sum((true_labels[:, i] == 0) & (pred_labels.iloc[:, i] == 0))
                    FP = np.sum((true_labels[:, i] == 0) & (pred_labels.iloc[:, i] == 1))
                    FN = np.sum((true_labels[:, i] == 1) & (pred_labels.iloc[:, i] == 0))

                    # compute metrics and store them
                    metrics = compute_metrics(TP, TN, FP, FN)
                    metrics["TP"] = TP
                    metrics["TN"] = TN
                    metrics["FP"] = FP
                    metrics["FN"] = FN
                    trials_metrics[label].append(metrics)

            # save metrics for each trial and compute averages
            for label in class_labels:
                per_trial_metrics_path = os.path.join(output_dir, f"metrics_per_trial_{label}.csv")
                pd.DataFrame(trials_metrics[label]).to_csv(per_trial_metrics_path, index=False)

               
                avg_metrics = pd.DataFrame(trials_metrics[label]).mean(axis=0).to_dict()
                avg_metrics_path = os.path.join(output_dir, f"avg_metrics_{label}.csv")
                pd.DataFrame([avg_metrics]).to_csv(avg_metrics_path, index=False)

if __name__ == "__main__":
    input_folders = [
        "/workspace/my_auxiliary_persistent/retrain_roentgen/generated_test_sets/results/results_test_fully_synthetic/",
        "/workspace/my_auxiliary_persistent/retrain_roentgen/generated_test_sets/results/results_test_upsampled/",
        "/workspace/my_auxiliary_persistent/upsampled_test_set/results_ground_truth/",
        "/workspace/my_auxiliary_persistent/upsampled_test_set/results_test_real/",
        "/workspace/my_auxiliary_persistent/retrain_roentgen/make_sure_i_have_them_results/test_on_real_test_set_40-80/",
        "/workspace/my_auxiliary_persistent/retrain_roentgen/make_sure_i_have_them_results/test_on_synthetic_test_set_40-80/",
    ]
    
    output_base = "/workspace/my_auxiliary_persistent/retrain_roentgen/generated_test_sets/confusion_matrices_accompanying_stats/"
    process_folders(input_folders, output_base)
