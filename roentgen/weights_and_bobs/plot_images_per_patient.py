import os
import pandas as pd
import matplotlib.pyplot as plt

# Directory to save the plots
output_directory = "/workspace/my_auxiliary_persistent/dataset_stats_plain/"
os.makedirs(output_directory, exist_ok=True)

# Function to plot and save the distribution of images per patient
def plot_images_per_patient(data, save_path):
    """
    Plot the distribution of images per patient.
    Parameters:
        data (str or pd.DataFrame): File path to the CSV or the DataFrame itself.
        save_path (str): Path to save the resulting plot.
    """
    # Load the CSV if a path is provided
    if isinstance(data, str):
        df = pd.read_csv(data)
    else:
        df = data

    # Extract patient IDs from the 'Path' column
    df['Patient ID'] = df['Path'].str.extract(r'(patient\d+)')

    # Count the number of images per patient
    image_counts = df['Patient ID'].value_counts()

    # Plot the distribution
    plt.figure(figsize=(10, 6))
    plt.hist(image_counts, bins=range(1, image_counts.max() + 2), edgecolor='black', align='left')
    plt.title("Distribution of Images per Patient")
    if isinstance(data, str):
        plt.title(f"Distribution of Images per Patient\n{os.path.basename(data)}")
    plt.xlabel("Number of Images per Patient")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    #plt.ylim(0, 30)
    
    # limit x axis 
    plt.xlim(0, 25)

    # Save the plot
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

# File paths
csv_files = {
    "ground_truth": "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/ground_truth.csv",
    "train": "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/train.csv",
    "valid": "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/valid.csv",
    "test": "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/test.csv",
}

# Generate plots for individual CSV files
for name, path in csv_files.items():
    plot_path = os.path.join(output_directory, f"{name}_images_per_patient.png")
    plot_images_per_patient(path, plot_path)

# Generate a combined dataset plot (in memory, without saving the combined CSV)
combined_df = pd.concat([pd.read_csv(path) for path in csv_files.values()], ignore_index=True)

# Plot for the combined dataset
combined_plot_path = os.path.join(output_directory, "combined_images_per_patient.png")
plot_images_per_patient(combined_df, combined_plot_path)

print(f"Plots have been saved to: {output_directory}")


