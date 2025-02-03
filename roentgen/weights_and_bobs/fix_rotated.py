# Import the necessary libraries
import os
from PIL import Image
import pandas as pd

#rotate and save images
def rotate_and_save_image(image_path, rotation):
    try:
        # open the image
        img = Image.open(image_path)
        # rotate the image based on the specified operation
        if rotation == 'rotate clockwise 90 degrees':
            img = img.rotate(-90, expand=True)
            operation_suffix = "_rot_90"
        elif rotation == 'rotate counter-clockwise 90 degrees':
            img = img.rotate(90, expand=True)
            operation_suffix = "_rot_-90"
        # new filename
        base, ext = os.path.splitext(image_path)
        new_image_path = base + operation_suffix + ext
        print("new_image_path: " + str(new_image_path))
        img.save(new_image_path)
        return new_image_path
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# operations and corresponding image paths
'''
operations = [
    ("rotate clockwise 90 degrees", "/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0_batch_2_train_1/patient18710/study1/view2_frontal.jpg"),
    ("rotate clockwise 90 degrees", "/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0_batch_3_train_2/patient29388/study2/view2_frontal.jpg"),
    ("rotate counter-clockwise 90 degrees", "/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0_batch_3_train_2/patient27714/study2/view1_frontal.jpg"),
    ("rotate clockwise 90 degrees", "/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0_batch_3_train_2/patient27078/study2/view1_frontal.jpg"),
    ("rotate counter-clockwise 90 degrees", "/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0_batch_4_train_3/patient50095/study1/view2_lateral.jpg"),
    ("rotate clockwise 90 degrees", "/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0_batch_2_train_1/patient16447/study10/view1_frontal.jpg"),
    ("rotate clockwise 90 degrees", "/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0_batch_2_train_1/patient06987/study8/view1_frontal.jpg"),
    ("rotate clockwise 90 degrees", "/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0_batch_3_train_2/patient25355/study5/view1_frontal.jpg"),
    ("rotate clockwise 90 degrees", "/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0_batch_3_train_2/patient32444/study9/view1_frontal.jpg"),
    ("rotate clockwise 90 degrees", "/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0_batch_2_train_1/patient07228/study1/view1_frontal.jpg"),
    ("rotate counter-clockwise 90 degrees", "/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0_batch_4_train_3/patient43555/study1/view1_frontal.jpg"),
    ("rotate counter-clockwise 90 degrees", "/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0_batch_4_train_3/patient60348/study4/view2_lateral.jpg"),
    ("rotate clockwise 90 degrees", "/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0_batch_2_train_1/patient09242/study7/view1_frontal.jpg"),
    ("rotate counter-clockwise 90 degrees", "/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0_batch_3_train_2/patient24059/study5/view2_frontal.jpg"),
    ("rotate clockwise 90 degrees", "/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0_batch_2_train_1/patient16958/study5/view1_frontal.jpg"),
    ("rotate counter-clockwise 90 degrees", "/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0_batch_2_train_1/patient09244/study2/view1_frontal.jpg"),
    ("rotate clockwise 90 degrees", "/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0_batch_2_train_1/patient17448/study2/view2_frontal.jpg"),
    ("rotate counter-clockwise 90 degrees", "/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0_batch_3_train_2/patient25679/study2/view1_frontal.jpg"),
    ("rotate clockwise 90 degrees", "/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0_batch_2_train_1/patient17448/study2/view1_frontal.jpg"),
    ("rotate counter-clockwise 90 degrees", "/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0_batch_3_train_2/patient39787/study5/view1_frontal.jpg"),
]
'''
#operations = [
#        ("rotate counter-clockwise 90 degrees", "/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0_batch_4_train_3/patient60348/study4/view1_frontal.jpg"),
#        ("rotate clockwise 90 degrees", "/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0_batch_2_train_1/patient06987/study8/view2_frontal.jpg"),
#]


# new operations start here for retraining roentgen
# train operations
'''
operations = [
        ("rotate counter-clockwise 90 degrees", "/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0_batch_3_train_2/patient34156/study1/view2_frontal.jpg"),
        ("rotate clockwise 90 degrees", "/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0_batch_2_train_1/patient04820/study1/view2_frontal.jpg"),
        ("rotate clockwise 90 degrees", "/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0_batch_4_train_3/patient57461/study2/view1_frontal.jpg"),
        ("rotate counter-clockwise 90 degrees", "/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0_batch_2_train_1/patient08690/study1/view1_frontal.jpg"),
        ("rotate clockwise 90 degrees", "/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0_batch_4_train_3/patient47893/study2/view1_frontal.jpg"),
        ("rotate counter-clockwise 90 degrees", "/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0_batch_2_train_1/patient01007/study6/view2_lateral.jpg"), # title num 35
        ("rotate counter-clockwise 90 degrees", "/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0_batch_3_train_2/patient34156/study1/view1_frontal.jpg"),
        ("rotate clockwise 90 degrees", "/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0_batch_3_train_2/patient27281/study3/view1_frontal.jpg"), # title num 52
        ("rotate clockwise 90 degrees", "/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0_batch_2_train_1/patient06169/study1/view1_frontal.jpg"),
        ("rotate clockwise 90 degrees", "/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0_batch_4_train_3/patient60483/study1/view1_frontal.jpg"), # title num 61
        ("rotate clockwise 90 degrees", "/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0_batch_2_train_1/patient08747/study9/view1_frontal.jpg"),
        ("rotate clockwise 90 degrees", "/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0_batch_4_train_3/patient48591/study2/view2_frontal.jpg"), # title num 80
        ("rotate clockwise 90 degrees", "/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0_batch_4_train_3/patient51867/study2/view1_frontal.jpg"), # title num 91
        ("rotate clockwise 90 degrees", "/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0_batch_3_train_2/patient29539/study1/view1_frontal.jpg"), # title num 94
        ("rotate counter-clockwise 90 degrees", "/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0_batch_4_train_3/patient60424/study1/view2_frontal.jpg"), # title num 97
        ("rotate clockwise 90 degrees", "/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0_batch_2_train_1/patient15999/study1/view1_frontal.jpg"), # title num 114
        ("rotate clockwise 90 degrees", "/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0_batch_3_train_2/patient31419/study2/view2_frontal.jpg"), # title num 122
        ("rotate counter-clockwise 90 degrees", "/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0_batch_3_train_2/patient28798/study6/view2_frontal.jpg"), # title num 124
        ("rotate clockwise 90 degrees", "/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0_batch_4_train_3/patient60424/study1/view1_frontal.jpg")
]
'''

# valid operations
operations = [
    ("rotate counter-clockwise 90 degrees", "/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0_batch_3_train_2/patient25951/study2/view1_frontal.jpg"),
    ("rotate clockwise 90 degrees", "/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0_batch_3_train_2/patient27281/study3/view2_frontal.jpg"),
]

print("num images to rotate: " + str(len(operations)))

#csv_path = "/workspace/persistent/code/roentgen/weights_and_bobs/diffusion_splits/processed_valid_with_projection.csv" # OLD
#csv_path = "/workspace/my_auxiliary_persistent/retrain_roentgen/csvs/diffusion_splits/processed_train_with_projection.csv" # NEW retrain roentgen on test
csv_path = "/workspace/my_auxiliary_persistent/retrain_roentgen/csvs/diffusion_splits/processed_valid_with_projection.csv" # NEW retrain roentgen on test
df = pd.read_csv(csv_path)
 # NEW retrain roentgen on test
num_rotated_images = 0
# iterate through the operations and save the rotated images
for rotation, image_path in operations:
    print("old image path: " + str(image_path))
    new_image_path = rotate_and_save_image(image_path, rotation)
    if new_image_path:
        df.loc[df['file_name'] == image_path, 'file_name'] = new_image_path
        print("haha ")
        num_rotated_images += 1

# save the updated CSV file with a new name
new_csv_path = csv_path.replace(".csv", "_corr_rot.csv")
print("new_csv_path: " + str(new_csv_path))
print("num images rotated successfully: " + str(num_rotated_images))
df.to_csv(new_csv_path, index=False)

