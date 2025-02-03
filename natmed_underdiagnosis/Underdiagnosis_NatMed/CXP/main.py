import torch
from classification.train import train
from classification.predictions import make_pred_multilabel
import pandas as pd


#PATH_TO_IMAGES = "/PATH TO DATASET IMAGES IN YOUR SERVER/datasets/CheXpert"
#PATH_TO_IMAGES = "/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/" # ORIGINALLY USED FOR TRAINING, ORIGINAL REAL WORLD IMAGES
#PATH_TO_IMAGES = "/workspace/my_auxiliary_persistent/roentgen_datasets/hugging_out_5_part4_ground_truth/checkpoint-25500/" # diffusion trials # checkpoints to do 19500 done, 25500 done, 30500, done, 34500 missing files.
#PATH_TO_IMAGES = "/workspace/my_auxiliary_persistent/roentgen_datasets/hugging_out_5_part3_test_set/checkpoint-25500/" # 1000 done, 19500 done, 25500 done, 30500 done, 34500 done, 4500 missing files

#TRAIN_DF_PATH ="/PATH TO DATASET CSV FILES IN YOUR SERVER/CheXpert/split/new_train.csv"
#TEST_DF_PATH ="/PATH TO DATASET CSV FILES IN YOUR SERVER/CheXpert/split/new_test.csv"
#VAL_DF_PATH = "/PATH TO DATASET CSV FILES IN YOUR SERVER/CheXpert/split/new_valid.csv"

#TRAIN_DF_PATH = "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/train.csv"
#TEST_DF_PATH = "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/ground_truth.csv"
#TEST_DF_PATH = "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/test.csv"

#VAL_DF_PATH = "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/valid.csv"

################ dataframe paths for training the attribute classifier on generated data ###########
# so split your diffusion genertated dataset into train, val and test
TRAIN_DF_PATH = "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_3/gt_attr_train.csv"
TEST_DF_PATH = "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_3/gt_attr_test.csv"  # test the test split
#TEST_DF_PATH = "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_3/refit_for_attribute_classifier_ground_truth.csv" # test on the whole (potentially real world) ground truth set
VAL_DF_PATH = "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_3/gt_attr_val.csv"


# retrain roentgen images attr classifier
#TRAIN_DF_PATH = "/workspace/my_auxiliary_persistent/retrain_roentgen/attribute_classifier/test_attr_train(in).csv"
#VAL_DF_PATH = "/workspace/my_auxiliary_persistent/retrain_roentgen/attribute_classifier/test_attr_val(in).csv"
#TEST_DF_PATH = "/workspace/my_auxiliary_persistent/retrain_roentgen/attribute_classifier/test_attr_test(in).csv"

#PATH_TO_IMAGES = "/workspace/my_auxiliary_persistent/retrain_roentgen/full_test_set_for_metric_assessment/checkpoint-44000/" # latest refinetuned roentgen
PATH_TO_IMAGES = "/workspace/my_auxiliary_persistent/retrain_roentgen/full_gt_set_for_metric_assessment/checkpoint-44000"

# more elaborate attri classifier
#TRAIN_DF_PATH = "/workspace/my_auxiliary_persistent/retrain_roentgen/attribute_classifier_part_2/test_attr_train.csv"
#VAL_DF_PATH = "/workspace/my_auxiliary_persistent/retrain_roentgen/attribute_classifier_part_2/test_attr_val.csv"
#TEST_DF_PATH = "/workspace/my_auxiliary_persistent/retrain_roentgen/attribute_classifier_part_2/test_attr_test.csv"

################# quick multicheckpoint eval sets ########################
#TEST_DF_PATH = "/workspace/my_auxiliary_persistent/retrain_roentgen/eval_checkpoints/filtered_test.csv"
#PATH_TO_IMAGES = "/workspace/my_auxiliary_persistent/retrain_roentgen/test_checkpoints_images/checkpoint-75500/"


################### experiments for training/evaluating classifier on test set (both real test and synthetic test set and mix of real and synth in test) ####################
#TRAIN_DF_PATH = "/workspace/my_auxiliary_persistent/retrain_roentgen/train_on_test/test_split_to_train_and_valid/train_split.csv" # train portion of test set
#TRAIN_DF_PATH = "/workspace/my_auxiliary_persistent/retrain_roentgen/train_on_test/real_and_synth_data_together_csvs/full_real_plus_full_synth.csv" # real + synthetic train portions of test set combined. DATASET.PY MODIFIED WHEN USING THIS!!!!!!!!!!!!!!!!!!
#TRAIN_DF_PATH = "/workspace/my_auxiliary_persistent/retrain_roentgen/train_on_test/real_and_synth_data_together_csvs/full_real_plus_half_synth.csv" # full real + half as many synthetic (stratified) 
#TRAIN_DF_PATH = "/workspace/my_auxiliary_persistent/retrain_roentgen/train_on_test/real_and_synth_data_together_csvs/full_real_plus_three_quarters_synth.csv" # full real + 3/4 as many synthetic (stratified)
#TRAIN_DF_PATH = "/workspace/my_auxiliary_persistent/retrain_roentgen/train_on_test/real_and_synth_data_together_csvs/full_real_plus_one_quarter_synth.csv" # full real + 1/4 as many synthetic (stratified)
#TRAIN_DF_PATH = "/workspace/my_auxiliary_persistent/retrain_roentgen/train_on_test/real_and_synth_data_together_csvs/full_real_plus_one_and_a_quarter_synthetic.csv" # full real + 1.25 as many synthetic (1 is same as test and additional 0.25 is stratified sample from test csv)
#TRAIN_DF_PATH = "/workspace/my_auxiliary_persistent/retrain_roentgen/train_on_test/real_and_synth_data_together_csvs/full_real_plus_one_and_two_quarters_synthetic.csv" # full real test set + 1.5 as many synthetic. 
#TRAIN_DF_PATH = "/workspace/my_auxiliary_persistent/retrain_roentgen/train_on_test/real_and_synth_data_together_csvs/full_real_plus_one_and_three_quarters_synthetic.csv" # full real test set + 1.75 as many synthetic
#VAL_DF_PATH = "/workspace/my_auxiliary_persistent/retrain_roentgen/train_on_test/test_split_to_train_and_valid/valid_split.csv" # valid portion of test set
#TEST_DF_PATH = "/workspace/persistent/code/natmed_underdiagnosis/Underdiagnosis_NatMed/CXP/my_splits_2/valid.csv" # For these epxeriments, real world valid set is test set
#PATH_TO_IMAGES = "/workspace/my_auxiliary_persistent/retrain_roentgen/full_test_set_for_metric_assessment/checkpoint-44000/" # retrained roentgen test set images

# We mix all existing data of the provoder regardless of their original validation/train label in the original dataset and split them into 80-10-10 train test and validation sets based on Patient-ID such that no patient images appears in more than one split. 


def main():

    MODE = "train"#"train"  # Select "train", "test", or "resume"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    val_df = pd.read_csv(VAL_DF_PATH)
    val_df_size = len(val_df)
    print("validation_df size:",val_df_size)

    train_df = pd.read_csv(TRAIN_DF_PATH)
    train_df_size = len(train_df)
    print("train_df size:", train_df_size)
    
    test_df = pd.read_csv(TEST_DF_PATH)
    test_df_size = len(test_df)
    print("test_df size:", test_df_size)


    if MODE == "train":
        modeltype = "densenet"  # currently code is based on densenet121 # ORIGINAL
        #modeltype = "resnet" # NEW, SHOULD NOT BE USING THIS USUALLY!!!
        #modeltype = "inception" # NEW, SHOULD NOT BE USING THIS USUALLY!!!
        #modeltype = "mobilenet" # NEW, SHOULD NOT BE USING THIS USUALLY!!!
        CRITERION = 'BCELoss'
        lr = 5e-4

        model, best_epoch = train(train_df, val_df, PATH_TO_IMAGES, modeltype, CRITERION, device,lr)


    if MODE =="test":
        val_df = pd.read_csv(VAL_DF_PATH)
        test_df = pd.read_csv(TEST_DF_PATH)
        
        # ORIGINAL
        #CheckPointData = torch.load('results/checkpoint') # GPU Available
        # LOAD SPECIFIC CHECKPOINT
        CheckPointData = torch.load('my_results_batch_size_192/checkpoint') # GPU Available
        #CheckPointData = torch.load('results/checkpoint', map_location=torch.device('cpu')) # no gpu available
        model = CheckPointData['model']

        make_pred_multilabel(model, test_df, val_df, PATH_TO_IMAGES, device)


    if MODE == "resume":
        modeltype = "resume" 
        CRITERION = 'BCELoss'
        lr = 0.1e-3

        model, best_epoch = train(train_df, val_df, PATH_TO_IMAGES, modeltype, CRITERION, device,lr)

        PlotLearnignCurve()


if __name__ == "__main__":
    main()
