import os
import pandas as pd
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix

#  function to compute metrics from confusion matrix components
def compute_metrics(TP, TN, FP, FN):
    metrics = {}
    metrics["accuracy"] = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
    metrics["precision"] = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    metrics["recall"] = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    metrics["specificity"] = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    metrics["f1_score"] = 2 * (metrics["precision"] * metrics["recall"]) / (metrics["precision"] + metrics["recall"]) if (metrics["precision"] + metrics["recall"]) > 0 else 0.0
    metrics["FPR"] = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    metrics["FNR"] = FN / (FN + TP) if (FN + TP) > 0 else 0.0
    metrics["prevalence"] = (TP + FN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
    metrics["P"] = TP + FN
    metrics["N"] = TN + FP
    return metrics

# process folders and compute confusion matrix and metrics
def process_single_trial_folders(input_folders, output_base):
    class_labels = [
        "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema", 
        "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other", 
        "Fracture", "Support Devices"
    ]

    for folder in input_folders:
        folder_name = os.path.basename(os.path.normpath(folder))
        output_dir = os.path.join(output_base, folder_name)
        os.makedirs(output_dir, exist_ok=True)

        true_labels_path = os.path.join(folder, "True.csv")
        pred_labels_path = os.path.join(folder, "bipred.csv")

        if not os.path.exists(true_labels_path) or not os.path.exists(pred_labels_path):
            print(f"Missing files in folder: {folder}")
            continue

        
        true_labels_df = pd.read_csv(true_labels_path)
        pred_labels_df = pd.read_csv(pred_labels_path)

        # drop 'Path' column if it exists
        true_labels = true_labels_df.drop(columns=["Path"]).values
        pred_labels = pred_labels_df.drop(columns=["Path"])

        # convert True/False to 1.0/0.0 in predicted labels
        pred_labels.replace({True: 1.0, False: 0.0}, inplace=True)

        # compute multilabel confusion matrix
        cm = multilabel_confusion_matrix(true_labels, pred_labels.values, labels=range(len(class_labels)))

        # Save confusion matrix to CSV
        cm_output_path = os.path.join(output_dir, "multilabel_confusion_matrix.csv")
        pd.DataFrame(cm.reshape(len(class_labels), -1), columns=["TN", "FP", "FN", "TP"], index=class_labels).to_csv(cm_output_path)

        #  metrics for each class
        metrics_per_class = []
        for i, label in enumerate(class_labels):
            TN, FP, FN, TP = cm[i].ravel()
            metrics = compute_metrics(TP, TN, FP, FN)
            metrics["class"] = label
            metrics_per_class.append(metrics)

        # save metrics for each class to CSV
        metrics_output_path = os.path.join(output_dir, "metrics_per_class.csv")
        pd.DataFrame(metrics_per_class).to_csv(metrics_output_path, index=False)

        # overall metrics across all classes
        overall_TP = cm[:, 1, 1].sum()
        overall_TN = cm[:, 0, 0].sum()
        overall_FP = cm[:, 0, 1].sum()
        overall_FN = cm[:, 1, 0].sum()
        overall_metrics = compute_metrics(overall_TP, overall_TN, overall_FP, overall_FN)

        # save as csv
        overall_metrics_output_path = os.path.join(output_dir, "overall_metrics.csv")
        pd.DataFrame([overall_metrics]).to_csv(overall_metrics_output_path, index=False)

if __name__ == "__main__":
    input_folders = [
        "/workspace/my_auxiliary_persistent/base_downstream_classifier_results_on_ground_truth_set/",
        "/workspace/my_auxiliary_persistent/base_downstream_classifier_results_on_test_set/"
    ]

    output_base = "/workspace/my_auxiliary_persistent/base_downstream_classifier_gt_confusion_matrix_stats/"
    process_single_trial_folders(input_folders, output_base)
