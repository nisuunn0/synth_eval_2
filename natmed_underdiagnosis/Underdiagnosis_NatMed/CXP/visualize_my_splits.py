import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


train_df = pd.read_csv("my_splits_2/train.csv")
#train_df = pd.read_csv("/workspace/my_auxiliary_persistent/retrain_roentgen/csvs/diffusion_splits/train_from_test.csv")
valid_df = pd.read_csv("my_splits_2/valid.csv")
#valid_df = pd.read_csv("/workspace/my_auxiliary_persistent/retrain_roentgen/csvs/diffusion_splits/valid_from_test.csv")
test_df = pd.read_csv("my_splits_2/test.csv")
ground_truth_df = pd.read_csv("my_splits_2/ground_truth.csv")


output_dir = "plots_2"
#output_dir = "/workspace/my_auxiliary_persistent/retrain_roentgen/csvs/split_stats_plots/"
os.makedirs(output_dir, exist_ok=True)


sex_order = ["Male", "Female"]
age_group_order = ["0-20", "20-40", "40-80", "80+"]


def plot_distribution(df, title, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()


    sns.countplot(x="Sex", data=df, ax=axes[0], order=sex_order)
    axes[0].set_title("Sex Distribution")


    sns.countplot(x="Age Group", data=df, ax=axes[1], order=age_group_order)
    axes[1].set_title("Age Group Distribution")


    sns.countplot(y="Lung Lesion", data=df, ax=axes[2])
    axes[2].set_title("Lung Lesion Distribution")
    #axes[2].set_ylabel("")
    
    sns.countplot(y="Pleural Effusion", data=df, ax=axes[3])
    axes[3].set_title("Pleural Effusion Distribution")

    
    #sns.countplot(y="Edema", data=df, ax=axes[3])
    #axes[3].set_title("Edema Distribution")


    #fig.delaxes(axes[3])

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()

    plt.savefig(output_path)
        
    plt.close()



def plot_all_distributions(train_df, valid_df, test_df, ground_truth_df, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # Plot distributions for each set
    plot_distribution(train_df, "Train Set", os.path.join(output_path, "train_distribution.jpg"))
    plot_distribution(valid_df, "Validation Set", os.path.join(output_path, "valid_distribution.jpg"))
    plot_distribution(test_df, "Test Set", os.path.join(output_path, "test_distribution.jpg"))
    plot_distribution(ground_truth_df, "Ground Truth Set", os.path.join(output_path, "ground_truth_distribution.jpg"))

  
    plt.tight_layout()

  
    plt.savefig(os.path.join(output_path, "all_distributions.jpg"))
    plt.close()


    
def pie_plot():
   
    train_count = len(train_df)
    valid_count = len(valid_df)
    test_count = len(test_df)
    ground_truth_count = len(ground_truth_df)
    
    print("trainountc: " + str(train_count))
    print("validountc: " + str(valid_count))
    print("testountc: " + str(test_count))
    print("gt: " + str(ground_truth_count))
    
    tot_num_rows = train_count + valid_count + test_count + ground_truth_count

 
    train_fraction = train_count / tot_num_rows
    valid_fraction = valid_count / tot_num_rows
    test_fraction = test_count / tot_num_rows
    ground_truth_fraction = ground_truth_count / tot_num_rows


    labels = ['Train', 'Valid', 'Test', 'Ground Truth']
    sizes = [train_fraction, valid_fraction, test_fraction, ground_truth_fraction]

  
    counts = [train_count, valid_count, test_count, ground_truth_count]

  
    def func(pct, allsizes, count_list):
        # the count_list already contains the raw counts, so just access them directly
        idx = int(pct / 100. * sum(allsizes))  # find the idx based on the percentage
        return f"{count_list[idx]} ({pct:.1f}%)"

 
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct=lambda pct: func(pct, sizes, counts), startangle=140)
    plt.axis('equal')  
    plt.title('Fractions of Rows in Each DataFrame')
    plt.savefig("/workspace/my_auxiliary_persistent/dataset_stats_plain/fractions_pie_chart_with_raw_counts.jpg")  # Save output as JPG


    
pie_plot()


#plot_distribution(train_df, "Train Set", os.path.join(output_dir, "train_distribution.jpg"))
#plot_distribution(valid_df, "Validation Set", os.path.join(output_dir, "valid_distribution.jpg"))
#plot_distribution(test_df, "Test Set", os.path.join(output_dir, "test_distribution.jpg"))
#plot_distribution(ground_truth_df, "Ground Truth Set", os.path.join(output_dir, "ground_truth_distribution.jpg"))


#plot_all_distributions(train_df, valid_df, test_df, ground_truth_df, output_dir)

