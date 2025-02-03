import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix

# plot and save confusion matrix
def plot_confusion_matrix(cm, class_labels, output_path):
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

# compute and save the average confusion matrix
def compute_and_save_avg_cm(trials_true_labels, trials_pred_labels, output_dir, class_labels):
    # init an empty array to accumulate confusion matrices
    multi_cm = np.zeros((len(class_labels), 2, 2))  # Shape will be (n_labels, 2, 2)

    #loop through the true and predicted labels for each trial and compute multilabel confusion matrix
    for true_labels, pred_labels in zip(trials_true_labels, trials_pred_labels):
        cm = multilabel_confusion_matrix(true_labels, pred_labels, labels=range(len(class_labels)))
        multi_cm += cm

    # average confusion matrix by averaging along axis 0
    multi_cm_avg = multi_cm / len(trials_true_labels)

    # flatten the confusion matrices for saving
    avg_cm_output_path = os.path.join(output_dir, "avg_multilabel_confusion_matrix.csv")
    pd.DataFrame(multi_cm_avg.reshape(len(class_labels), -1), columns=["TN", "FP", "FN", "TP"], index=class_labels).to_csv(avg_cm_output_path)

    # plot and save the average multilabel confusion matrix as an image
    #avg_cm_plot_path = os.path.join(output_dir, "avg_multilabel_confusion_matrix.png")
    #plot_confusion_matrix(multi_cm_avg.mean(axis=0), class_labels, avg_cm_plot_path)


def process_folders(input_folders, output_base):
    class_labels = [
        "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema", 
        "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other", 
        "Fracture", "Support Devices"
    ]
    
    for folder in input_folders:
        folder_name = os.path.basename(os.path.normpath(folder))  # Extract the general folder name
        for subfolder in os.listdir(folder):
            subfolder_path = os.path.join(folder, subfolder)
            print("subfolder path: ################: " + str(subfolder_path))
            if not os.path.isdir(subfolder_path):
                continue

            # create the output directory structure including the general folder name 
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

               
                true_labels = true_labels_df.drop(columns=["Path"]).values
                pred_labels = pred_labels_df.drop(columns=["Path"])

                
                print(f"\nBefore Conversion - Trial {trial}:")
                print(pred_labels_df.head())

                # Ensure that we are only converting boolean True/False to 1.0/0.0
                pred_labels.replace({True: 1.0, False: 0.0}, inplace=True)

               
                print(f"\nAfter Conversion to 1.0/0.0 - Trial {trial}:")
                print(pred_labels.head())

                # store the true labels and predicted labels for each trial
                trials_true_labels.append(true_labels)
                trials_pred_labels.append(pred_labels.values)

                #confusion matrix for this trial (multilabel)
                cm = multilabel_confusion_matrix(true_labels, pred_labels.values, labels=range(len(class_labels)))

                # save confusion matrix as CSV
                cm_output_path = os.path.join(output_dir, f"multilabel_confusion_matrix_trial_{trial}.csv")
                pd.DataFrame(cm.reshape(len(class_labels), -1), columns=["TN", "FP", "FN", "TP"], index=class_labels).to_csv(cm_output_path)

                # plot and save as img
                #cm_plot_path = os.path.join(output_dir, f"multilabel_confusion_matrix_trial_{trial}.png")
                #plot_confusion_matrix(cm.mean(axis=0), class_labels, cm_plot_path)

            # After processing all trials, compute and save the average multilabel confusion matrix
            compute_and_save_avg_cm(trials_true_labels, trials_pred_labels, output_dir, class_labels)

if __name__ == "__main__":
    input_folders = [
        #"/workspace/my_auxiliary_persistent/retrain_roentgen/generated_test_sets/results/results_test_fully_synthetic/",
        #"/workspace/my_auxiliary_persistent/retrain_roentgen/generated_test_sets/results/results_test_upsampled/",
        #"/workspace/my_auxiliary_persistent/upsampled_test_set/results_ground_truth/",
        #"/workspace/my_auxiliary_persistent/upsampled_test_set/results_test_real/",
        "/workspace/my_auxiliary_persistent/retrain_roentgen/make_sure_i_have_them_results/test_on_real_test_set_40-80/",
        "/workspace/my_auxiliary_persistent/retrain_roentgen/make_sure_i_have_them_results/test_on_synthetic_test_set_40-80/",
    ]
    
    output_base = "/workspace/my_auxiliary_persistent/retrain_roentgen/generated_test_sets/confusion_matrices_and_related_stats/"
    process_folders(input_folders, output_base)
