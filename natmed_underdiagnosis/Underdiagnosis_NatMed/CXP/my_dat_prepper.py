import pandas as pd
import os
from tqdm import tqdm

#persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0_batch_3_train_2

RESULT_CSV_PATH = "/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/train_cheXbert-myCopy1.csv"
VALID_PATH = DATASET_PATH + "/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0_batch_1_validate_&_csv"
TRAIN_PATHS = [
    "/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0_batch_2_train_1", 
    "/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0_batch_3_train_2", 
    "/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0_batch_4_train_3"
]
#persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/train_cheXbert-myCopy1.csv
def update_image_paths(csv_path, train_paths):
    # Read the CSV file
    df = pd.read_csv(csv_path)
       
    # OLD WORKING 
    '''
    def update_image_path(row):
        for folder in train_paths:
            #print("folder: " + folder)
            base_folder = os.path.basename(folder)
            #print("base_folder: " + base_folder)
            # Remove the redundant "/CheXpert-v1.0/train/" part from the path
            path_without_train = row['Path'].replace(f"CheXpert-v1.0/train/", "")
            #print("path_without_train: " + path_without_train)
            full_path = os.path.join(folder, path_without_train)
            #print("full_path: " + str(full_path))
            if os.path.exists(full_path):
                # Uncomment the next line to see what paths would be updated
                print(f"Updating path from {row['Path']} to {full_path}")
                return full_path
        return row['Path']
    '''

    # NEW TEST
    def update_image_path(row):
        for folder in train_paths:
            base_folder = os.path.basename(folder)
            # remove the redundant "/CheXpert-v1.0/train/" part from the path
            path_without_train = row['Path'].replace(f"CheXpert-v1.0/train/", "")
            full_path = os.path.join(folder, path_without_train)
            if os.path.exists(full_path):
                # get the relevant part of the path after the base folder
                relevant_part = full_path.split(base_folder, 1)[1].lstrip('/')
                # final path including the base folder
                final_path = os.path.join(base_folder, relevant_part)
                #print(f"pdating path from {row['Path']} to {final_path}")
                return final_path
        return row['Path']




    tqdm.pandas(desc="Updating paths")
    #df = df.loc[190000:190020]#df.head(10)
    df['Path'] = df.progress_apply(update_image_path, axis=1)


    df.to_csv(csv_path, index=False)

def do_it_all():
    print("curr dir: " + os.getcwd())
    update_image_paths(RESULT_CSV_PATH, TRAIN_PATHS)

if __name__ == "__main__":
    do_it_all()

