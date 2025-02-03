folder natmed:_underdiagnosis/Underdiagnosis_NatMed/CXP/ contains scripts related to the downstream classifer, particularly its training scripts and evaluation routines, given in main.py and main_multi_trial_stats.py respectively.
This folder is based on and a modified variant of the github repository found in: https://github.com/LalehSeyyed/Underdiagnosis_NatMed 

folder roentgen/weights_and_bobs/ contains scripts that pertain to inference and attempts at finetuneing roentgen (the scripts for finetuneing RoentGen in this folder do not work and are attempts). It also contains dataset preparation routines. Files like infer_roentgen_test_sets.py and its variants are used to generate synthetic CXR images according to input csv files.

For further finetuneing roentgen, fine_tune_roentgen_hugging_face/train_txt_to_img.py was used. This script is a modified version of the original script to enable also trianing the text encoder along the U-Net. This script relies on the HuggingFace diffusers library/environment as seen here: https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py


No dataset specific files (e.g., csv files) nor any model weights are stored in this repository, only code.
