import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# File path
csv_file_path = "/workspace/my_auxiliary_persistent/retrain_roentgen/consistency_test_subpop_level_summary/consistency_subpop_level_summary.csv"
output_image_path = "/workspace/my_auxiliary_persistent/retrain_roentgen/consistency_test_subpop_level_summary/model_performance_by_age_group.png"

# Read the CSV file
df = pd.read_csv(csv_file_path)

# Extract the Age Group column and all the model columns
age_groups = df["Age Group"]
model_columns = df.columns[1:]

# Parse AUC and CI values from the "mean±ci" format
parsed_data = {}
for column in model_columns:
    auc_values = []
    ci_values = []
    for value in df[column]:
        mean, ci = value.split("±")
        auc_values.append(float(mean))
        ci_values.append(float(ci))
    parsed_data[column] = {"AUC": np.array(auc_values), "CI": np.array(ci_values)}

# Plotting
fig, ax = plt.subplots(figsize=(14, 8))

# Bar width and positions
bar_width = 0.1
x = np.arange(len(age_groups))  # X-axis positions for the age groups
group_offset = 0  # Offset to group bars for each age group

# Colors for each model
colors = plt.cm.tab20.colors  # Use a colormap for consistent, visually distinct colors

# Plot each model's bars
for i, column in enumerate(model_columns):
    auc_values = parsed_data[column]["AUC"]
    ci_values = parsed_data[column]["CI"]

    # Compute the positions of bars within each age group
    bar_positions = x + group_offset
    ax.bar(
        bar_positions,
        auc_values,
        width=bar_width,
        label=column,
        color=colors[i % len(colors)],
        yerr=ci_values,
        capsize=3, #4
        alpha=0.9,
        #error_kw={'linewidth': 0.5},  # error bar width
    )

    # Increment group offset for the next model
    group_offset += bar_width

# Customize the x-axis
ax.set_xticks(x + (group_offset - bar_width) / 2)
ax.set_xticklabels(age_groups, rotation=45, ha="right")

# Set y-axis range
ax.set_ylim(0.73, 0.87)

# Add labels and legend
ax.set_xlabel("Age Group")
ax.set_ylabel("AUC")
ax.set_title("Model Performance by Age Group")
ax.legend(title="Models", bbox_to_anchor=(1.05, 1), loc="upper left")

# Add grid lines for clarity
ax.grid(axis="y", linestyle="--", alpha=0.7)

# Adjust layout
plt.tight_layout()

# Save the plot as an image
plt.savefig(output_image_path, dpi=300)
print(f"Plot saved successfully to {output_image_path}")
