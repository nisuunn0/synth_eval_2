import os
import pandas as pd
import numpy as np
from classification.predictions_multitrial_update import make_pred_multilabel
import torch

PATH_TO_IMAGES = "/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/" # ORIGINAL REAL WORLD IMAGES
#PATH_TO_IMAGES = "/workspace/my_auxiliary_persistent/roentgen_datasets/hugging_out_5_part4_ground_truth/checkpoint-25500/" # DIFFUSION GENERATED IMAGES ground truth
#PATH_TO_IMAGES = "/workspace/my_auxiliary_persistent/roentgen_datasets/hugging_out_5_part3_test_set/checkpoint-25500/" # DIFFUSION GENERATED IMAGES test set
#PATH_TO_IMAGES = "/workspace/my_auxiliary_persistent/upsampled_test_set/imgs/" # test set with upsampled synthetic and fully synthetic images
#PATH_TO_IMAGES = "/workspace/my_auxiliary_persistent/retrain_roentgen/generated_test_sets/imgs/" # refinetuned roentgen
#PATH_TO_IMAGES = "/workspace/my_auxiliary_persistent/retrain_roentgen/full_test_set_for_metric_assessment/checkpoint-44000/" # refinetuned roentgen on full synthetic test set
#PATH_TO_IMAGES = "/workspace/my_auxiliary_persistent/retrain_roentgen/full_gt_set_for_metric_assessment/checkpoint-44000" # latest finetuned roentgen full gt set
#VAL_DF_PATH = "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/valid.csv" # usually use this more or less i guess

#VAL_DF_PATH = "/workspace/my_auxiliary_persistent/retrain_roentgen/train_on_test/test_split_to_train_and_valid/valid_split.csv" # for the classifier trained on the test set (I guess for validation and getting threshold results you should use the val split portion of the test set)

VAL_DF_PATH = "/workspace/my_auxiliary_persistent/retrain_roentgen/attribute_classifier_part_2/test_attr_val.csv" # used to get threshold for attribute classifier
#TEST_DF_PATH = "/workspace/my_auxiliary_persistent/retrain_roentgen/attribute_classifier_part_2/test_attr_test.csv"
#VAL_DF_PATH = "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_3/gt_attr_val.csv"

# generate unique filenames based on CSV name and trial number
def generate_result_filename(base_dir, csv_file, trial_num):
    base_filename = os.path.splitext(os.path.basename(csv_file))[0]
    trial_dir = os.path.join(base_dir, base_filename)
    
    os.makedirs(trial_dir, exist_ok=True)
    
    return os.path.join(trial_dir, f"trial_{trial_num}.csv")

# Function to calculate mean and confidence intervals for a list of values
def calculate_statistics(results):
    """
    Given a list of results (e.g., mean AUCs or AUPRCs), compute the mean and confidence intervals.
    """
    mean = np.mean(results)
    std_dev = np.std(results)
    conf_interval = 1.96 * std_dev / np.sqrt(len(results))  # 95% confidence interval
    return mean, conf_interval

def run_multiple_tests(csv_files, num_trials):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #CheckPointData = torch.load('my_results_batch_size_192/checkpoint') # GPU Available # This is the default classifier you trained, in general you should be using this
    #CheckPointData = torch.load('/workspace/my_auxiliary_persistent/retrain_roentgen/train_on_test/results/train_on_real_test/train_out/checkpoint') # this model was trained on real world test set
    #CheckPointData = torch.load('/workspace/my_auxiliary_persistent/retrain_roentgen/train_on_test/results/train_on_synth_test/train_out/checkpoint') # this model was trained on synthetic test set
    #CheckPointData = torch.load('/workspace/my_auxiliary_persistent/retrain_roentgen/train_on_test/results/train_on_real_and_synth_test/train_out/checkpoint') # this model was trained on real and synthetic test set (real test set + synthetic test set)
    #CheckPointData = torch.load('/workspace/my_auxiliary_persistent/retrain_roentgen/train_on_test/results/train_on_full_real_and_half_synth_test/train_out/checkpoint') # this model was trained on full real and half as many synthetic test set (full real test set + half as many synthetic test set)
    #CheckPointData = torch.load('/workspace/my_auxiliary_persistent/retrain_roentgen/train_on_test/results/train_on_full_real_and_three_quarters_synth_test/train_out/checkpoint') # this model was trained on full real + 3/4 as many synthetic test set.
    #CheckPointData = torch.load('/workspace/my_auxiliary_persistent/retrain_roentgen/train_on_test/results/train_on_full_real_and_one_quarter_synth_test/train_out/checkpoint') # this model was trained on full real + 1/4 as many synthetic test
    #CheckPointData = torch.load('/workspace/my_auxiliary_persistent/retrain_roentgen/train_on_test/results/train_on_full_real_and_one_and_a_quarter_synth_test/train_out/checkpoint') # this model was trained on full real + 1.25 as many synthetic test
    #CheckPointData = torch.load('/workspace/my_auxiliary_persistent/retrain_roentgen/train_on_test/results/train_on_full_real_and_one_and_two_quarters_synth_test/train_out/checkpoint') # this model was trained on full real + 1.5 as many synthetic test (1.25 is images from previous training, this one has 0.25 new)
    #CheckPointData = torch.load('/workspace/my_auxiliary_persistent/retrain_roentgen/train_on_test/results/train_on_full_real_and_one_and_three_quarters_synth_test/train_out/checkpoint') # this model was trained on full real + 1.75 as many synthetic test
    
    ############ consistency tests checkpoints #################
    #CheckPointData = torch.load('/workspace/my_auxiliary_persistent/consistency_tests/resnet50_train_out/checkpoint') # resnet50 trained on regular train set
    #CheckPointData = torch.load('/workspace/my_auxiliary_persistent/consistency_tests/inceptionv3_train_out/checkpoint') # inception
    #CheckPointData = torch.load('/workspace/my_auxiliary_persistent/consistency_tests/mobilenetv3_train_out/checkpoint') # mobilenetv3
    
    ############### attri classifier checkpoints ###############################
    #CheckPointData = torch.load('/workspace/my_auxiliary_persistent/retrain_roentgen/attribute_classifier_part_2/train_out_for_trained_on_real/checkpoint')
    CheckPointData = torch.load('/workspace/my_auxiliary_persistent/retrain_roentgen/attribute_classifier_part_2/train_out_for_trained_on_synth/checkpoint')
    #CheckPointData = torch.load('/workspace/my_auxiliary_persistent/retrain_roentgen/attribute_classifier_part_2/trained_on_real_gt_train_out/checkpoint')
    #CheckPointData = torch.load('/workspace/my_auxiliary_persistent/retrain_roentgen/attribute_classifier_part_2/trained_on_synth_gt_train_out/checkpoint')
    
    model = CheckPointData['model']
    
    val_df = pd.read_csv(VAL_DF_PATH)
    #base_output_dir = "/workspace/my_auxiliary_persistent/natmed_multitrial_results/" # inference on real world data out dir
    #base_output_dir = "/workspace/my_auxiliary_persistent/natmed_multitrial_results_synthetic_data/" # inference on diffusion generated data out dir
    #base_output_dir = "/workspace/my_auxiliary_persistent/upsampled_test_set/results_test_upsampled/" # inference on test set upsampled with synthetic data
    #base_output_dir = "/workspace/my_auxiliary_persistent/upsampled_test_set/results_test_fully_synthetic/" # inference on test set made up of fully synthetic set
    #base_output_dir = "/workspace/my_auxiliary_persistent/upsampled_test_set/results_test_real/" # inference on real world test set (50 trials per subgroup)
    #base_output_dir = "/workspace/my_auxiliary_persistent/upsampled_test_set/results_ground_truth/" # inference on real world ground truth set (50 trials per subgroup)
    
    #base_output_dir = "/workspace/my_auxiliary_persistent/retrain_roentgen/generated_test_sets/results/results_test_upsampled/"
    #base_output_dir = "/workspace/my_auxiliary_persistent/retrain_roentgen/generated_test_sets/results/results_test_fully_synthetic/"
    #base_output_dir = "/workspace/my_auxiliary_persistent/retrain_roentgen/generated_test_sets/results/results_test_missing_portion/" # test set 40-80 # already exists elsewhere too? As in, I think this is just real world test set results, that already exist?
    
    #base_output_dir = "/workspace/my_auxiliary_persistent/natmed_multi_trial_results_2/" # missing portion of test results (majority age group 40-80) for the wrongly trained diffusion model on the valid set.
    #base_output_dir = "/workspace/my_auxiliary_persistent/natmed_multi_trial_results_3/" # missing portion of test results (majority age group 40-80) real world data
    
    # train on test set and test on valid set experiments
    #base_output_dir = "/workspace/my_auxiliary_persistent/retrain_roentgen/train_on_test/results/train_on_real_test/multi_trial_results/"
    #base_output_dir = "/workspace/my_auxiliary_persistent/retrain_roentgen/train_on_test/results/train_on_synth_test/multi_trial_results/"
    #base_output_dir = "/workspace/my_auxiliary_persistent/retrain_roentgen/train_on_test/results/train_on_real_and_synth_test/multi_trial_results/"
    #base_output_dir = "/workspace/my_auxiliary_persistent/retrain_roentgen/train_on_test/results/train_on_full_real_and_half_synth_test/multi_trial_results/"
    #base_output_dir = "/workspace/my_auxiliary_persistent/retrain_roentgen/train_on_test/results/train_on_full_real_and_three_quarters_synth_test/multi_trial_results/"
    #base_output_dir = "/workspace/my_auxiliary_persistent/retrain_roentgen/train_on_test/results/train_on_full_real_and_one_quarter_synth_test/multi_trial_results/"
    #base_output_dir = "/workspace/my_auxiliary_persistent/retrain_roentgen/train_on_test/results/train_on_full_real_and_one_and_a_quarter_synth_test/multi_trial_results/"
    #base_output_dir = "/workspace/my_auxiliary_persistent/retrain_roentgen/train_on_test/results/train_on_full_real_and_one_and_two_quarters_synth_test/multitrial_results/"
    #base_output_dir = "/workspace/my_auxiliary_persistent/retrain_roentgen/train_on_test/results/train_on_full_real_and_one_and_three_quarters_synth_test/multi_trial_results/"
    
    
    ################ consistency tests output dirs #####################
    #base_output_dir = "/workspace/my_auxiliary_persistent/consistency_tests/resnet50_results_synthetic/"
    #base_output_dir = "/workspace/my_auxiliary_persistent/consistency_tests/inceptionv3_results_synthetic/"
    #base_output_dir = "/workspace/my_auxiliary_persistent/consistency_tests/densenet121_results_synthetic/"
    #base_output_dir = "/workspace/my_auxiliary_persistent/consistency_tests/densenet121_results_real_world/"
    #base_output_dir = "/workspace/my_auxiliary_persistent/consistency_tests/mobilenetv3_results_synthetic/"
    
    ############## ensure i have trials (you almost certainly already ran these and have these, but just running them in case I missed/lost them)
    #base_output_dir = "/workspace/my_auxiliary_persistent/retrain_roentgen/make_sure_i_have_them_results/test_on_synthetic_test_set/"
    #base_output_dir = "/workspace/my_auxiliary_persistent/retrain_roentgen/make_sure_i_have_them_results/test_on_real_test_set/"
    
    #################### double check 80+_female upsample results (80+ female results for upsample multitrial look shockingly good, so double checking) #################
    #base_output_dir = "/workspace/my_auxiliary_persistent/retrain_roentgen/generated_test_sets/double_check_80+_female_upsample_results"
    #base_output_dir = "/workspace/my_auxiliary_persistent/retrain_roentgen/generated_test_sets/double_check_80+_female_upsample_results/and_the_other_csv/"
    
    ##################### base downstream classifier multitrial stats on entire test set and entire ground truth set #################################
    #base_output_dir = "/workspace/my_auxiliary_persistent/base_downstream_classifier_multi_trial_results/"
    
    ################ consistency tests on subpop level ######################
    #base_output_dir = "/workspace/my_auxiliary_persistent/retrain_roentgen/consistency_test_subpop_level/resnet_real/"
    #base_output_dir = "/workspace/my_auxiliary_persistent/retrain_roentgen/consistency_test_subpop_level/resnet_synth/"
    #base_output_dir = "/workspace/my_auxiliary_persistent/retrain_roentgen/consistency_test_subpop_level/mobilenet_real/"
    #base_output_dir = "/workspace/my_auxiliary_persistent/retrain_roentgen/consistency_test_subpop_level/mobilenet_synth/"
    #base_output_dir = "/workspace/my_auxiliary_persistent/retrain_roentgen/consistency_test_subpop_level/inception_real/"
    #base_output_dir = "/workspace/my_auxiliary_persistent/retrain_roentgen/consistency_test_subpop_level/inception_synth/"
    #base_output_dir = "/workspace/my_auxiliary_persistent/retrain_roentgen/consistency_test_subpop_level/densenet_synth/"
    #base_output_dir = "/workspace/my_auxiliary_persistent/retrain_roentgen/consistency_test_subpop_level/densenet_real/"
    
    ############### consistency models evaluation on subpops #######################
    #base_output_dir = "/workspace/my_auxiliary_persistent/additional_model_multi_trial_stats/resnet_ground_truth/"
    #base_output_dir = "/workspace/my_auxiliary_persistent/additional_model_multi_trial_stats/resnet_test_upsampled/"
    #base_output_dir = "/workspace/my_auxiliary_persistent/additional_model_multi_trial_stats/resnet_test_fully_synthetic/"
    #base_output_dir = "/workspace/my_auxiliary_persistent/additional_model_multi_trial_stats/inception_ground_truth/"
    #base_output_dir = "/workspace/my_auxiliary_persistent/additional_model_multi_trial_stats/inception_test_fully_synthetic/"
    #base_output_dir = "/workspace/my_auxiliary_persistent/additional_model_multi_trial_stats/inception_test_upsampled/"
    #base_output_dir = "/workspace/my_auxiliary_persistent/additional_model_multi_trial_stats/mobilenet_test_upsampled/"
    #base_output_dir = "/workspace/my_auxiliary_persistent/additional_model_multi_trial_stats/mobilenet_test_fully_synthetic/"
    #base_output_dir = "/workspace/my_auxiliary_persistent/additional_model_multi_trial_stats/mobilenet_ground_truth/"
    
    
    ################################ attri classifier #################################
    #base_output_dir = "/workspace/my_auxiliary_persistent/retrain_roentgen/attribute_classifier_part_2/trained_on_real_multi_trial_stats_for_real_world_set/"
    #base_output_dir = "/workspace/my_auxiliary_persistent/retrain_roentgen/attribute_classifier_part_2/trained_on_real_multi_trial_stats_for_synth_set/"
    #base_output_dir = "/workspace/my_auxiliary_persistent/retrain_roentgen/attribute_classifier_part_2/trained_on_synth_multi_trial_stats_for_synth_set/"
    #base_output_dir = "/workspace/my_auxiliary_persistent/retrain_roentgen/attribute_classifier_part_2/trained_on_synth_muylti_trial_stats_for_real_world_set/"
    #base_output_dir = "/workspace/my_auxiliary_persistent/retrain_roentgen/attribute_classifier_part_2/trained_on_real_gt_multi_trial_stats_real_gt/"
    #base_output_dir = "/workspace/my_auxiliary_persistent/retrain_roentgen/attribute_classifier_part_2/trained_on_real_gt_multi_trial_stats_synth_gt/"
    #base_output_dir = "/workspace/my_auxiliary_persistent/retrain_roentgen/attribute_classifier_part_2/trained_on_synth_gt_multi_trial_stats_synth_gt/"
    #base_output_dir = "/workspace/my_auxiliary_persistent/retrain_roentgen/attribute_classifier_part_2/trained_on_synth_gt_multi_trial_stats_real_gt/"
    #base_output_dir = "/workspace/my_auxiliary_persistent/retrain_roentgen/attribute_classifier_part_2/trained_on_synth_gt_multi_trial_stats_real_gt_subpops/"
    #base_output_dir = "/workspace/my_auxiliary_persistent/retrain_roentgen/attribute_classifier_part_2/trained_on_synth_gt_multi_trial_stats_real_whole_gt/"
    #base_output_dir = "/workspace/my_auxiliary_persistent/retrain_roentgen/attribute_classifier_part_2/trained_on_real_gt_multi_trial_stats_synth_whole_gt/"
    #base_output_dir = "/workspace/my_auxiliary_persistent/retrain_roentgen/attribute_classifier_part_2/trained_on_real_test_multi_trial_stats_for_synth_whole_test/"
    base_output_dir = "/workspace/my_auxiliary_persistent/retrain_roentgen/attribute_classifier_part_2/trained_on_synth_test_multi_trial_stats_for_whole_real_world_test/"
    
    for csv_file in csv_files:
        # Load the test set CSV
        test_df = pd.read_csv(csv_file)

        # Store the average AUC and AUPRC results for all trials
        auc_results = []
        auprc_results = []
        
        # Initialize dictionary to store disease-specific AUC and AUPRC results
        disease_specific_results = {}
        
        for trial_num in range(num_trials):
            # Determine the sample size based on the size of the test dataset
            
            '''
            dataset_size = len(test_df)
            # Apply different sampling rates based on dataset size
            if dataset_size <= 4000:
                sample_size = dataset_size  # Use the full dataset (100%)
            elif 4000 < dataset_size <= 8000:
                sample_size = int(0.7 * dataset_size)  # Use 70% of the dataset
            elif 8000 < dataset_size <= 16000:
                sample_size = int(0.4 * dataset_size)  # Use 40% of the dataset
            else:
                sample_size = int(0.25 * dataset_size)  # Use 30% of the dataset
            '''
            
            # Sample the test set with replacement (bootstrapping)
            #test_df_sampled = test_df.sample(n=sample_size, replace=True).reset_index(drop=True)
            test_df_sampled = test_df.sample(n=len(test_df), replace=True).reset_index(drop=True)
            #test_df_sampled = test_df.sample(n=16, replace=True).reset_index(drop=True)  # FOR TESTING ONLY, LOAD SMALL NUM SAMPLES
            
            base_filename = os.path.splitext(os.path.basename(csv_file))[0]
            
            # Create directory and filename for saving results
            result_filename = generate_result_filename(base_output_dir, csv_file, trial_num)
            
            # Run the test on the sampled dataset and get the TestEval_df
            _, _, TestEval_df = make_pred_multilabel(
                model, test_df_sampled, val_df, PATH_TO_IMAGES, device,
                output_dir=os.path.dirname(result_filename), trial_num=trial_num
            )
            
            # Calculate average AUC and AUPRC for this trial
            avg_auc = TestEval_df['auc'].mean()
            avg_auprc = TestEval_df['auprc'].mean()

            # Collect average AUC and AUPRC results across trials
            auc_results.append(avg_auc)
            auprc_results.append(avg_auprc)
            
            # Collect disease-specific AUC and AUPRC
            for index, row in TestEval_df.iterrows():
                disease = row['label']
                if disease not in disease_specific_results:
                    disease_specific_results[disease] = {'auc': [], 'auprc': []}
                disease_specific_results[disease]['auc'].append(row['auc'])
                disease_specific_results[disease]['auprc'].append(row['auprc'])

            # Save the TestEval_df with a unique filename for this trial
            TestEval_df.to_csv(result_filename, index=False)
            print(f"Saved trial {trial_num} results to {result_filename}")

        # After all trials, calculate the mean and confidence interval for AUC and AUPRC
        mean_auc, auc_conf_interval = calculate_statistics(auc_results)
        mean_auprc, auprc_conf_interval = calculate_statistics(auprc_results)

        # Save the summary statistics for this CSV
        summary_filename = os.path.join(base_output_dir, base_filename, f"{base_filename}_summary.csv")
        summary_df = pd.DataFrame({
            "Metric": ["Mean AUC", "Mean AUPRC"],
            "Mean": [mean_auc, mean_auprc],
            "Confidence Interval": [auc_conf_interval, auprc_conf_interval]
        })
        summary_df.to_csv(summary_filename, index=False)
        print(f"Saved summary statistics for {csv_file} to {summary_filename}")

        # Optionally: Calculate and save disease-specific statistics
        disease_summary = []
        for disease, metrics in disease_specific_results.items():
            mean_auc_disease, auc_conf_interval_disease = calculate_statistics(metrics['auc'])
            mean_auprc_disease, auprc_conf_interval_disease = calculate_statistics(metrics['auprc'])
            disease_summary.append({
                'Disease': disease,
                'Mean AUC': mean_auc_disease,
                'AUC Confidence Interval': auc_conf_interval_disease,
                'Mean AUPRC': mean_auprc_disease,
                'AUPRC Confidence Interval': auprc_conf_interval_disease
            })

        # Convert to DataFrame and save disease-specific summary
        disease_summary_df = pd.DataFrame(disease_summary)
        disease_summary_filename = os.path.join(base_output_dir, base_filename, f"{base_filename}_disease_summary.csv")
        disease_summary_df.to_csv(disease_summary_filename, index=False)
        print(f"Saved disease-specific summary statistics for {csv_file} to {disease_summary_filename}")

# Function to retrieve all CSV files from a directory
def get_csv_files_from_directory(directory_path):
    # List all files in the directory
    all_files = os.listdir(directory_path)
    
    # Filter to get only CSV files
    csv_files = [os.path.join(directory_path, f) for f in all_files if f.endswith('.csv')]
    
    return csv_files


# List of CSV files to test
#csv_files = [
#    "test_data_1.csv",
#    "test_data_2.csv",
    # Add more CSV file paths here
#]


# Specify the directory where your CSV files are located
csv_directory = "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/"

# Get the list of ALL csv files in directory
#csv_files = get_csv_files_from_directory(csv_directory)

'''
# only ground truth
csv_files = ["/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/ground_truth.csv", "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/ground_truth_80+.csv",
            "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/ground_truth_80+_Male.csv", "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/ground_truth_80+_Female.csv",
            "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/ground_truth_20-40.csv", "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/ground_truth_20-40_Male.csv",
            "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/ground_truth_20-40_Female.csv", "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/ground_truth_0-20.csv",
            "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/ground_truth_0-20_Male.csv", "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/ground_truth_0-20_Female.csv",]
'''

'''
# ground truth subpops only
csv_files = ["/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/ground_truth_80+.csv",
            "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/ground_truth_80+_Male.csv", "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/ground_truth_80+_Female.csv",
            "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/ground_truth_20-40.csv", "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/ground_truth_20-40_Male.csv",
            "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/ground_truth_20-40_Female.csv", "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/ground_truth_0-20.csv",
            "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/ground_truth_0-20_Male.csv", "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/ground_truth_0-20_Female.csv",]
'''

'''
# ground truth continued from gpu restart
csv_files = ["/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/ground_truth_20-40_Female.csv", "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/ground_truth_0-20.csv",
            "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/ground_truth_0-20_Male.csv", "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/ground_truth_0-20_Female.csv",]
'''

'''
# test set (real world images)
csv_files = ["/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/test.csv", "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/test_80+.csv", 
            "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/test_80+_Male.csv", "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/test_80+_Female.csv",
            "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/test_40-80.csv", "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/test_40-80_Male.csv",
            "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/test_40-80_Female.csv", "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/test_20-40.csv",
            "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/test_20-40_Male.csv", "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/test_20-40_Female.csv",
            "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/test_0-20.csv", "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/test_0-20_Male.csv",
            "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/test_0-20_Female.csv",]
'''

'''
# test upsampled with synthetic
csv_files = ["/workspace/my_auxiliary_persistent/upsampled_test_set/csv_files/test_combined_with_upsamples/test_0-20_Female.csv", "/workspace/my_auxiliary_persistent/upsampled_test_set/csv_files/test_combined_with_upsamples/test_0-20_Male.csv",
            "/workspace/my_auxiliary_persistent/upsampled_test_set/csv_files/test_combined_with_upsamples/test_0-20.csv", "/workspace/my_auxiliary_persistent/upsampled_test_set/csv_files/test_combined_with_upsamples/test_20-40_Female.csv",
            "/workspace/my_auxiliary_persistent/upsampled_test_set/csv_files/test_combined_with_upsamples/test_20-40_Male.csv", "/workspace/my_auxiliary_persistent/upsampled_test_set/csv_files/test_combined_with_upsamples/test_20-40.csv",
            "/workspace/my_auxiliary_persistent/upsampled_test_set/csv_files/test_combined_with_upsamples/test_80+_Female.csv", "/workspace/my_auxiliary_persistent/upsampled_test_set/csv_files/test_combined_with_upsamples/test_80+_Male.csv",
            "/workspace/my_auxiliary_persistent/upsampled_test_set/csv_files/test_combined_with_upsamples/test_80+.csv"]
'''

'''
# missing test portion for the wrongly trained roentgen on validation
# you also should run these csvs through the retrained roentgen
csv_files = ["/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/test_40-80.csv", "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/test_40-80_Male.csv",
            "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/test_40-80_Female.csv"]
'''

'''
# test made up of only synthetic samples
csv_files = ["/workspace/my_auxiliary_persistent/upsampled_test_set/csv_files/synth_test_set/test_0-20_Female_to_be_added_upsamples.csv", "/workspace/my_auxiliary_persistent/upsampled_test_set/csv_files/synth_test_set/test_0-20_Male_to_be_added_upsamples.csv", "/workspace/my_auxiliary_persistent/upsampled_test_set/csv_files/synth_test_set/test_0-20_to_be_added_upsamples.csv", "/workspace/my_auxiliary_persistent/upsampled_test_set/csv_files/synth_test_set/test_20-40_Female_to_be_added_upsamples.csv", "/workspace/my_auxiliary_persistent/upsampled_test_set/csv_files/synth_test_set/test_20-40_Male_to_be_added_upsamples.csv", "/workspace/my_auxiliary_persistent/upsampled_test_set/csv_files/synth_test_set/test_20-40_to_be_added_upsamples.csv", "/workspace/my_auxiliary_persistent/upsampled_test_set/csv_files/synth_test_set/test_80+_Female_to_be_added_upsamples.csv", "/workspace/my_auxiliary_persistent/upsampled_test_set/csv_files/synth_test_set/test_80+_Male_to_be_added_upsamples.csv", "/workspace/my_auxiliary_persistent/upsampled_test_set/csv_files/synth_test_set/test_80+_to_be_added_upsamples.csv"]
'''

# double checking 80+ female results (for retrained roentgen the 80+ female results looked surprisingly good, so double checking them
#csv_files = #["/workspace/my_auxiliary_persistent/upsampled_test_set/csv_files/test_combined_with_upsamples/test_80+_Female.csv",]#
#csv_files = ["/workspace/my_auxiliary_persistent/upsampled_test_set/csv_files/synth_test_set/test_80+_Female_to_be_added_upsamples.csv"]

# valid set as test set, used for training on real + synthetic
#csv_files = ["/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/valid.csv"]


# regular test set, used for consistency thus far
#csv_files = ["/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/test.csv"]

# gt + test
#csv_files = ["/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/test.csv", "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/ground_truth.csv"]

# consistency tests on subpop level. all subpop splits of test set
'''
csv_files = ["/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/test_80+.csv", 
            "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/test_80+_Male.csv", "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/test_80+_Female.csv",
            "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/test_40-80.csv", "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/test_40-80_Male.csv",
            "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/test_40-80_Female.csv", "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/test_20-40.csv",
            "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/test_20-40_Male.csv", "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/test_20-40_Female.csv",
            "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/test_0-20.csv", "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/test_0-20_Male.csv",
            "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/test_0-20_Female.csv"]
'''

'''
# ground truth attribute classifier age subgroups 
csv_files = ["/workspace/my_auxiliary_persistent/retrain_roentgen/attribute_classifier_part_2/gt_attr_splits/gt_0-20_split_all_sexes.csv",
            "/workspace/my_auxiliary_persistent/retrain_roentgen/attribute_classifier_part_2/gt_attr_splits/gt_0-20_split_female.csv",
            "/workspace/my_auxiliary_persistent/retrain_roentgen/attribute_classifier_part_2/gt_attr_splits/gt_0-20_split_male.csv",
            "/workspace/my_auxiliary_persistent/retrain_roentgen/attribute_classifier_part_2/gt_attr_splits/gt_20-40_split_all_sexes.csv",
            "/workspace/my_auxiliary_persistent/retrain_roentgen/attribute_classifier_part_2/gt_attr_splits/gt_20-40_split_female.csv",
            "/workspace/my_auxiliary_persistent/retrain_roentgen/attribute_classifier_part_2/gt_attr_splits/gt_20-40_split_male.csv",
            "/workspace/my_auxiliary_persistent/retrain_roentgen/attribute_classifier_part_2/gt_attr_splits/gt_80+_split_all_sexes.csv",
            "/workspace/my_auxiliary_persistent/retrain_roentgen/attribute_classifier_part_2/gt_attr_splits/gt_80+_split_female.csv",
            "/workspace/my_auxiliary_persistent/retrain_roentgen/attribute_classifier_part_2/gt_attr_splits/gt_80+_split_male.csv",]
'''

#csv_files = ["/workspace/my_auxiliary_persistent/retrain_roentgen/attribute_classifier_part_2/test_attr_test.csv"]
#csv_files = ["/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_3/gt_attr_test.csv"]
#csv_files = ["/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_3/refit_for_attribute_classifier_ground_truth.csv"]
csv_files = ["/workspace/my_auxiliary_persistent/retrain_roentgen/attribute_classifier_part_2/refit_for_attribute_classifier_test.csv"]
# to get definitive results, I guess just run attri classifier on full real or full synth gt or test set (e.g., trained on synth gt set, then test on full real gt set).

# number of trials to run per csv
#num_trials = 50 #20 # USUALLY USED 50 THUS Far
num_trials = 30 # consistency, 30 trials

# run em trials 
run_multiple_tests(csv_files, num_trials)

