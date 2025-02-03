import matplotlib.pyplot as plt

# Additional groups hardcoded data
additional_groups = [
    # Group 1: Female
    {
        "test_real": {"label": "test_real (n=9948)", "AUC": 0.811, "CI": 0.0009},
        "test_synthetic": {"label": "test_synthetic (n=9948)", "AUC": 0.831, "CI": 0.001},
        "age_sex": "40_80_Female"
    },
    # Group 2: Male
    {
        "test_real": {"label": "test_real (n=15412)", "AUC": 0.814, "CI": 0.0006},
        "test_synthetic": {"label": "test_synthetic (n=15412)", "AUC": 0.834, "CI": 0.0006},
        "age_sex": "40_80_Male"
    },
    # Group 3: General 40-80 group
    {
        "test_real": {"label": "test_real (n=25360)", "AUC": 0.813, "CI": 0.0004},
        "test_synthetic": {"label": "test_synthetic (n=25360)", "AUC": 0.832, "CI": 0.0006},
        "age_sex": "40_80"
    }
]


plt.figure(figsize=(10, 6)) 
x_positions = []
current_position = 0
gap_between_additional_groups = 3 
gap_between_bars_within_group = 0.7  
mean_aucs = []
auc_cis = []
additional_group_labels = []  


for group in additional_groups:
    # Plot "test_real"
    real_data = group["test_real"]
    plt.bar(current_position, real_data["AUC"], color="lightgreen", yerr=real_data["CI"], capsize=5, label=real_data["label"], edgecolor="green")
    x_positions.append(current_position)
    additional_group_labels.append(real_data["label"])  # Label for the 'test_real' bar
    mean_aucs.append(real_data["AUC"])
    auc_cis.append(real_data["CI"])
    current_position += gap_between_bars_within_group  # Adjust spacing for bars within a group

    # plot "test_synthetic"
    synthetic_data = group["test_synthetic"]
    plt.bar(current_position, synthetic_data["AUC"], color="orange", yerr=synthetic_data["CI"], capsize=5, label=synthetic_data["label"], edgecolor="darkorange")
    x_positions.append(current_position)
    additional_group_labels.append(synthetic_data["label"])  # Label for the 'test_synthetic' bar
    mean_aucs.append(synthetic_data["AUC"])
    auc_cis.append(synthetic_data["CI"])
    current_position += 1  # Move to the next group

    # add age and sex information above the bars
    group_center = (x_positions[-2] + x_positions[-1]) / 2
    plt.text(group_center, max(mean_aucs) + max(auc_cis) * 0.2, group["age_sex"], ha="center", fontsize=12, rotation=45)

    # increase the gap between groups
    current_position += gap_between_additional_groups


plt.xticks(x_positions, additional_group_labels, rotation=45, ha='right')


plt.xlabel('Experiment Groups')
plt.ylabel('Mean AUC')
#plt.title('Mean AUC for 40-80 Age Groups')
#plt.ylim(min(mean_aucs) - max(auc_cis), max(mean_aucs) + max(auc_cis) * 1.5)
plt.ylim(0.75, 0.9)
plt.tight_layout()


fig_out_dir = "/workspace/my_auxiliary_persistent/retrain_roentgen/generated_test_sets/plots_of_results/multi_trial_stats_40_80_only_inception.png"
plt.savefig(fig_out_dir)
