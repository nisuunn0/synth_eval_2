import pandas as pd

# Load your data
true_df = pd.read_csv("/workspace/my_auxiliary_persistent/retrain_roentgen/generated_test_sets/results/results_test_fully_synthetic/test_0-20_Female_to_be_added_upsamples/True_trial_0.csv")
pred_df = pd.read_csv('/workspace/my_auxiliary_persistent/retrain_roentgen/generated_test_sets/results/results_test_fully_synthetic/test_0-20_Female_to_be_added_upsamples/bipred_trial_0.csv')

# Remove 'bi_' prefix from the predicted labels
pred_df.columns = [col.replace('bi_', '') for col in pred_df.columns]

# Convert the boolean values to 1.0 (True) and 0.0 (False)
pred_df = pred_df.applymap(lambda x: 1.0 if x is True else (0.0 if x is False else x))

# Align the columns by Path index
true_df.set_index('Path', inplace=True)
pred_df.set_index('Path', inplace=True)

# Find mismatches directly by comparing columns
mismatches = true_df != pred_df

# Filter out the rows where any mismatch exists (i.e., True values in the 'mismatches' DataFrame)
mismatch_paths = mismatches[mismatches.any(axis=1)].index

# Print out the paths with mismatches
print("Paths with mismatches:")
print(mismatch_paths)

# Optional: If you want to see the specific mismatched values:
print("\nMismatched values:")
print(mismatches.loc[mismatch_paths])


