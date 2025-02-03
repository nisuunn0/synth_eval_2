# was used for combining csvs of additional 1/4 of synthetic test set to full real + full synth. So total is full real + 1.25 as many synthetic images from same distribution.

import pandas as pd

# Paths to the input CSVs and the output CSV. # these were used for full real test set + full synth test set + 1/4 synth test set
#first_csv_path = "/workspace/my_auxiliary_persistent/retrain_roentgen/train_on_test/real_and_synth_data_together_csvs/full_real_plus_full_synth.csv"
#second_csv_path = "/workspace/my_auxiliary_persistent/retrain_roentgen/train_on_test/additional_images/additional_samples_sampled_from_test.csv"
#output_csv_path = "/workspace/my_auxiliary_persistent/retrain_roentgen/train_on_test/real_and_synth_data_together_csvs/full_real_plus_one_and_a_quarter_synthetic.csv"

# these were used for full real test set + full synth test + 2/4 synth test set
#first_csv_path = "/workspace/my_auxiliary_persistent/retrain_roentgen/train_on_test/real_and_synth_data_together_csvs/full_real_plus_one_and_a_quarter_synthetic.csv"
#second_csv_path = "/workspace/my_auxiliary_persistent/retrain_roentgen/train_on_test/additional_images/additional_samples_sampled_from_test_part_2.csv"
#output_csv_path = "/workspace/my_auxiliary_persistent/retrain_roentgen/train_on_test/real_and_synth_data_together_csvs/full_real_plus_one_and_two_quarters_synthetic.csv"

# these were used for fukk reat test set + full synth test + 3/4 synth test set
first_csv_path = "/workspace/my_auxiliary_persistent/retrain_roentgen/train_on_test/real_and_synth_data_together_csvs/full_real_plus_one_and_two_quarters_synthetic.csv"
second_csv_path = "/workspace/my_auxiliary_persistent/retrain_roentgen/train_on_test/additional_images/additional_samples_sampled_from_test_part_3.csv"
output_csv_path = "/workspace/my_auxiliary_persistent/retrain_roentgen/train_on_test/real_and_synth_data_together_csvs/full_real_plus_one_and_three_quarters_synthetic.csv"

# synthetic image prefix
#synthetic_image_prefix = "/workspace/my_auxiliary_persistent/retrain_roentgen/train_on_test/additional_images/the_images/checkpoint-44000/"
#synthetic_image_prefix = "/workspace/my_auxiliary_persistent/retrain_roentgen/train_on_test/additional_images/the_images_2/checkpoint-44000/"
synthetic_image_prefix = "/workspace/my_auxiliary_persistent/retrain_roentgen/train_on_test/additional_images/the_images_3/checkpoint-44000/"

first_csv = pd.read_csv(first_csv_path)


second_csv = pd.read_csv(second_csv_path)

# update "Path" column in the second CSV to include the synthetic image prefix
second_csv["Path"] = second_csv["Path"].apply(lambda p: synthetic_image_prefix + p)

# combine em csvs
combined_csv = pd.concat([first_csv, second_csv], ignore_index=True)

# save
combined_csv.to_csv(output_csv_path, index=False)

print(f"Combined CSV saved to: {output_csv_path}")
