import pandas as pd
import matplotlib.pyplot as plt
import os

# Define directories and file names
#csv_dir = "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/"
#csv_dir = "/workspace/my_auxiliary_persistent/retrain_roentgen/train_on_test/test_split_to_train_and_valid/"
csv_dir = "/workspace/my_auxiliary_persistent/retrain_roentgen/train_on_test/real_and_synth_data_together_csvs/"
#output_dir = "/workspace/my_auxiliary_persistent/dataset_stats_plain/"
output_dir = "/workspace/my_auxiliary_persistent/retrain_roentgen/train_on_test/split_stats_plots/"
#csv_files = ["valid.csv", "train.csv", "test.csv", "ground_truth.csv"]
#csv_files = ["train_split.csv", "valid_split.csv"]
csv_files = ["full_real_plus_half_synth.csv"]

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Function to plot the class distribution and save the figure
def plot_and_save_class_distribution(file_path, file_name, output_dir):
    # Read the CSV file
    data = pd.read_csv(file_path)
    
    # Exclude the first and last columns
    class_columns = data.columns[5:-3]#-2] # usually -2, trying -3 when last column in csv is "stratifiy_col"
    
    # Calculate the sum for each class
    class_counts = data[class_columns].sum()
    
    # Plot the distribution
    plt.figure(figsize=(10, 6))
    class_counts.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title(f"Class Distribution in {file_name}")
    plt.xlabel("Classes")
    plt.ylabel("Quantity")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the figure
    save_path = os.path.join(output_dir, f"{file_name}_class_distribution.png")
    plt.savefig(save_path)
    plt.close()  # Close the figure to avoid overlaps

# Plot and save distribution for each file separately
for file_name in csv_files:
    file_path = os.path.join(csv_dir, file_name)
    plot_and_save_class_distribution(file_path, file_name, output_dir)

# Combine and plot the distribution for all files
combined_counts = None
for file_name in csv_files:
    file_path = os.path.join(csv_dir, file_name)
    data = pd.read_csv(file_path)
    class_columns = data.columns[5:-2]
    if combined_counts is None:
        combined_counts = data[class_columns].sum()
    else:
        combined_counts += data[class_columns].sum()

# Plot combined class distribution and save
plt.figure(figsize=(10, 6))
combined_counts.plot(kind='bar', color='lightcoral', edgecolor='black')
plt.title("Combined Class Distribution Across All Files")
plt.xlabel("Classes")
plt.ylabel("Quantity")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Save the combined figure
#combined_save_path = os.path.join(output_dir, "combined_class_distribution.png")
#plt.savefig(combined_save_path)
plt.close()
