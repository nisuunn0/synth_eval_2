import os
import pandas as pd
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix

# compute and save the average confusion matrix as ratios
def compute_and_save_avg_cm(trials_true_labels, trials_pred_labels, output_dir, class_labels):
    # init an empty array to accumulate confusion matrices
    multi_cm = np.zeros((len(class_labels), 2, 2))  # Shape will be (n_labels, 2, 2)

    # loop through the true and predicted labels for each trial and compute multilabel confusion matrix
    for true_labels, pred_labels in zip(trials_true_labels, trials_pred_labels):
        cm = multilabel_confusion_matrix(true_labels, pred_labels, labels=range(len(class_labels)))
        multi_cm += cm

    # calculate the average confusion matrix by averaging along axis 0
    multi_cm_avg = multi_cm / len(trials_true_labels)

    # calculate the ratios for the average confusion matrix
    cm_avg_ratios = np.zeros_like(multi_cm_avg, dtype=float)
    for i in range(len(class_labels)):
        FN = multi_cm_avg[i, 0, 1]  # False Negatives
        TP = multi_cm_avg[i, 1, 1]  # True Positives
        TN = multi_cm_avg[i, 0, 0]  # True Negatives
        FP = multi_cm_avg[i, 1, 0]  # False Positives

        total_true = FN + TP  # Total true instances for the class
        total_pred = TN + FP  # Total predicted instances for the class

        # avoid division by zero
        if total_true > 0:
            cm_avg_ratios[i, 0, 0] = TN / (TN + FP)  # True Negative Ratio (TNR)
            cm_avg_ratios[i, 0, 1] = FN / (FN + TP)  # False Negative Ratio (FNR)
            cm_avg_ratios[i, 1, 0] = FP / (TN + FP)  # False Positive Ratio (FPR)
            cm_avg_ratios[i, 1, 1] = TP / (FN + TP)  # True Positive Ratio (TPR)

    # save the average confusion matrix ratios as CSV
    avg_cm_output_path = os.path.join(output_dir, "avg_multilabel_confusion_matrix_ratios.csv")
    pd.DataFrame(cm_avg_ratios.reshape(len(class_labels), -1), columns=["TNR", "FNR", "FPR", "TPR"], index=class_labels).to_csv(avg_cm_output_path)

# plot and save confusion matrix
def plot_confusion_matrix(cm, class_labels, output_path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(cm, cmap='Blues')
    fig.colorbar(cax)

    ax.set_xticks(np.arange(len(class_labels)))
    ax.set_yticks(np.arange(len(class_labels)))

    ax.set_xticklabels(class_labels, rotation=90)
    ax.set_yticklabels(class_labels)

    plt.xlabel('Predicted')
    plt.ylabel('True')

    # annotate each cell with the numeric value
    for i in range(len(class_labels)):
        for j in range(len(class_labels)):
            ax.text(j, i, format(cm[i, j], '.2f'),
                    ha="center", va="center", color="black")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# function to process the folders and generate confusion matrices
def process_folders(input_folders, output_base):
    '''
    # usual classes
    class_labels = [
        "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema", 
        "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other", 
        "Fracture", "Support Devices"
    ]
    '''
    
    '''
    # attribute classifier 
    class_labels = [
    "Sex bin", "frontlat bin", "AP", "PA", "Age Group 0-20", "Age Group 20-40", "Age Group 80+"
]
    
    '''
    # attribute_classifier with majority 40-80
    class_labels = [
    "Sex bin", "frontlat bin", "AP", "PA", "Age Group 0-20", "Age Group 20-40", 
    "Age Group 40-80", "Age Group 80+"
]

    
    # Loop through each folder in the input directories
    for folder in input_folders:
        folder_name = os.path.basename(os.path.normpath(folder))  # Extract the general folder name
        for subfolder in os.listdir(folder):
            subfolder_path = os.path.join(folder, subfolder)
            print("subfolder path: ################: " + str(subfolder_path))
            if not os.path.isdir(subfolder_path):
                continue

            # create the output directory structure
            output_dir = os.path.join(output_base, folder_name, subfolder)
            print("output_dir##############: " + str(output_dir))
            os.makedirs(output_dir, exist_ok=True)

            trials_true_labels = []
            trials_pred_labels = []

            # loop through trials (0-49)
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

                # store the true labels and predicted labels for each trial
                trials_true_labels.append(true_labels)
                trials_pred_labels.append(pred_labels.values)

                # Compute confusion matrix for this trial (multilabel)
                cm = multilabel_confusion_matrix(true_labels, pred_labels.values, labels=range(len(class_labels)))

                # Calculate the ratios
                cm_ratios = np.zeros_like(cm, dtype=float)
                for i in range(len(class_labels)):
                    FN = cm[i, 0, 1]  # False Negatives
                    TP = cm[i, 1, 1]  # True Positives
                    TN = cm[i, 0, 0]  # True Negatives
                    FP = cm[i, 1, 0]  # False Positives
                    
                    total_true = FN + TP  # Total true instances for the class
                    
                    # avoid division by 0
                    if total_true > 0:
                        cm_ratios[i, 0, 0] = TN / (TN + FP)  # True Negative Ratio (TNR)
                        cm_ratios[i, 0, 1] = FN / (FN + TP)  # False Negative Ratio (FNR)
                        cm_ratios[i, 1, 0] = FP / (TN + FP)  # False Positive Ratio (FPR)
                        cm_ratios[i, 1, 1] = TP / (FN + TP)  # True Positive Ratio (TPR)

                # save confusion matrix as CSV (ratios now instead of raw counts)
                cm_output_path = os.path.join(output_dir, f"multilabel_confusion_matrix_ratios_trial_{trial}.csv")
                pd.DataFrame(cm_ratios.reshape(len(class_labels), -1), columns=["TNR", "FNR", "FPR", "TPR"], index=class_labels).to_csv(cm_output_path)

            # after processing all trials, compute and save the average multilabel confusion matrix (ratios)
            compute_and_save_avg_cm(trials_true_labels, trials_pred_labels, output_dir, class_labels)

if __name__ == "__main__":
    #  input folders
    '''
    input_folders = [
        "/workspace/my_auxiliary_persistent/retrain_roentgen/generated_test_sets/results/results_test_fully_synthetic/",
        "/workspace/my_auxiliary_persistent/retrain_roentgen/generated_test_sets/results/results_test_upsampled/",
        "/workspace/my_auxiliary_persistent/upsampled_test_set/results_ground_truth/",
        "/workspace/my_auxiliary_persistent/upsampled_test_set/results_test_real/",
        "/workspace/my_auxiliary_persistent/retrain_roentgen/make_sure_i_have_them_results/test_on_real_test_set_40-80/",
        "/workspace/my_auxiliary_persistent/retrain_roentgen/make_sure_i_have_them_results/test_on_synthetic_test_set_40-80/",
    ]
    '''
    
    '''
    # resnet ones
    input_folders = [
        "/workspace/my_auxiliary_persistent/additional_model_multi_trial_stats/resnet_test_fully_synthetic/",
        "/workspace/my_auxiliary_persistent/additional_model_multi_trial_stats/resnet_test_upsampled/",
        "/workspace/my_auxiliary_persistent/additional_model_multi_trial_stats/resnet_ground_truth/",
        "/workspace/my_auxiliary_persistent/retrain_roentgen/consistency_test_subpop_level/resnet_real/",
        "/workspace/my_auxiliary_persistent/retrain_roentgen/consistency_test_subpop_level/resnet_real_40-80/"
    ]
    '''
    
    '''
    input_folders = [
        "/workspace/my_auxiliary_persistent/retrain_roentgen/attribute_classifier_part_2/trained_on_real_gt_multi_trial_stats_real_gt/",
        "/workspace/my_auxiliary_persistent/retrain_roentgen/attribute_classifier_part_2/trained_on_real_gt_multi_trial_stats_synth_gt/",
        "/workspace/my_auxiliary_persistent/retrain_roentgen/attribute_classifier_part_2/trained_on_real_gt_multi_trial_stats_synth_whole_gt/",
        "/workspace/my_auxiliary_persistent/retrain_roentgen/attribute_classifier_part_2/trained_on_synth_gt_multi_trial_stats_real_gt/",
        "/workspace/my_auxiliary_persistent/retrain_roentgen/attribute_classifier_part_2/trained_on_synth_gt_multi_trial_stats_real_gt/",
        "/workspace/my_auxiliary_persistent/retrain_roentgen/attribute_classifier_part_2/trained_on_synth_gt_multi_trial_stats_real_whole_gt/",
        "/workspace/my_auxiliary_persistent/retrain_roentgen/attribute_classifier_part_2/trained_on_synth_gt_multi_trial_stats_synth_gt/", 
    ]
    '''
    
    input_folders = [
        "/workspace/my_auxiliary_persistent/retrain_roentgen/attribute_classifier_part_2/trained_on_synth_test_multi_trial_stats_for_whole_real_world_test/",
        "/workspace/my_auxiliary_persistent/retrain_roentgen/attribute_classifier_part_2/trained_on_synth_muylti_trial_stats_for_real_world_set/",
        "/workspace/my_auxiliary_persistent/retrain_roentgen/attribute_classifier_part_2/trained_on_synth_multi_trial_stats_for_synth_set/",
        "/workspace/my_auxiliary_persistent/retrain_roentgen/attribute_classifier_part_2/trained_on_real_test_multi_trial_stats_for_synth_whole_test/",
        "/workspace/my_auxiliary_persistent/retrain_roentgen/attribute_classifier_part_2/trained_on_real_multi_trial_stats_for_synth_set/",
        "/workspace/my_auxiliary_persistent/retrain_roentgen/attribute_classifier_part_2/trained_on_real_multi_trial_stats_for_real_world_set",
    
    ]
    
    # Output directory
    #output_base = "/workspace/my_auxiliary_persistent/retrain_roentgen/generated_test_sets/confusion_matrices_as_ratios/"
    #output_base = "/workspace/my_auxiliary_persistent/retrain_roentgen/ratio_confusion_matrices_latter_models/resnet/"
    #output_base = "/workspace/my_auxiliary_persistent/retrain_roentgen/attribute_classifier_part_2/confusion_matrices/trained_on_gt/"
    output_base = "/workspace/my_auxiliary_persistent/retrain_roentgen/attribute_classifier_part_2/confusion_matrices/trained_on_real/"


    process_folders(input_folders, output_base)
