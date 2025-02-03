import pandas as pd
import cv2
import numpy as np
from scipy.ndimage import convolve, rotate
from skimage.morphology import skeletonize
from skimage.feature import canny
from scipy.stats import norm
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def make_features(image):
    scale = 512
    R = image[:,:,0]  # use the red channel
    if R.shape[0] != scale or R.shape[1] != scale:
        Rx = cv2.resize(R, (scale, scale))
    else:
        Rx = R
    m, n = Rx.shape
    t = int(m * 0.2)
    
    # feature 1
    Rx_ver = Rx[:, t*2:t*3]
    Rx_hor = Rx[t*2:t*3, :]
    mu_ver = np.mean(Rx_ver)
    mu_hor = np.mean(Rx_hor)
    feat_1 = mu_ver > mu_hor
    
    # build kernel
    sigma = 1
    mu = 0
    length = 9
    theta = 30
    X = np.linspace(-sigma, sigma, length)
    f = norm.pdf(X, mu, sigma)
    line = np.ones(length)
    g = np.outer(line, f)
    g = np.pad(g, ((length//2, length//2), (0, 0)), mode='constant')
    g = rotate(g, theta, reshape=False)
    
    tam = 25
    edg_hor = canny(Rx_hor, sigma=0.05)
    edg_ver = canny(Rx_ver, sigma=0.05)
    
    # feature 2
    O_hor = convolve(edg_hor.astype(float), g, mode='reflect')
    E_hor = skeletonize(O_hor > np.percentile(O_hor, 99))
    
    O_ver = convolve(edg_ver.astype(float), g, mode='reflect')
    E_ver = skeletonize(O_ver > np.percentile(O_ver, 99))
    
    E_left_hor = np.sum(E_hor[:, :E_hor.shape[1]//2])
    E_right_hor = np.sum(E_hor[:, E_hor.shape[1]//2:])
    feat_2 = E_left_hor > E_right_hor
    
    # feature 3
    gr = rotate(g, 90, reshape=False)
    Or_ver = convolve(edg_ver.astype(float), gr, mode='reflect')
    Er_ver = skeletonize(Or_ver > np.percentile(Or_ver, 99))
    
    E_up_ver = np.sum(Er_ver[:Er_ver.shape[0]//2, :])
    E_down_ver = np.sum(Er_ver[Er_ver.shape[0]//2:, :])
    feat_3 = E_up_ver > E_down_ver
    
    # feature 4 and 5
    mv, nv = Rx_ver.shape
    mh, nh = Rx_hor.shape
    
    feat_4 = np.std(Rx_ver[:mv//2, :]) > np.std(Rx_ver[mv//2:, :])
    feat_5 = np.std(Rx_hor[:, :nh//2]) > np.std(Rx_hor[:, nh//2:])
    
    return [feat_1, feat_2, feat_3, feat_4, feat_5]

def process_image_range(image_paths, output_dir):
    results = []

    for image_path in tqdm(image_paths):
        image = cv2.imread(image_path)
        features = make_features(image)

        if features[0]:
            orientation = 'Upright'
        elif features[1]:
            orientation = 'Upside-Down'
        elif features[2]:
            orientation = 'Right-Rotated'
        else:
            orientation = 'Left-Rotated'
        
        if orientation != 'Upright':
            results.append({'image_path': image_path, 'orientation': orientation})
            
            # Construct the save path
            path_parts = image_path.split(os.sep)
            filename = '_'.join(path_parts[-3:]).replace(os.sep, '_')
            save_image_path = os.path.join(output_dir, filename)

            cv2.imwrite(save_image_path, image)
    
    return results

def process_images(csv_file, output_csv, output_dir, num_processes=None):
    df = pd.read_csv(csv_file)
    image_paths = df['file_name']

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    num_processes = num_processes or cpu_count()
    chunk_size = len(image_paths) // num_processes

    chunks = [image_paths[i:i + chunk_size] for i in range(0, len(image_paths), chunk_size)]

    with Pool(processes=num_processes) as pool:
        results = pool.starmap(process_image_range, [(chunk, output_dir) for chunk in chunks])

    # Flatten the list of lists
    results = [item for sublist in results for item in sublist]
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)

# Run the function with your CSV file
# OLD
#process_images('/workspace/persistent/code/roentgen/weights_and_bobs/diffusion_splits/processed_train_with_projection.csv',
#               '/workspace/persistent/code/roentgen/weights_and_bobs/orientation_results_train/results.csv', 
#               '/workspace/persistent/code/roentgen/weights_and_bobs/orientation_results_train/imgs',
#               16)

# NEW retrain roentgen on test
#process_images('/workspace/my_auxiliary_persistent/retrain_roentgen/csvs/diffusion_splits/processed_train_with_projection.csv',
#              '/workspace/my_auxiliary_persistent/retrain_roentgen/csvs/orientation_results_train/results.csv',
#              '/workspace/my_auxiliary_persistent/retrain_roentgen/csvs/orientation_results_train/imgs',
#              16)

# NEW retrain roentgen on test
process_images('/workspace/my_auxiliary_persistent/retrain_roentgen/csvs/diffusion_splits/processed_valid_with_projection.csv',
              '/workspace/my_auxiliary_persistent/retrain_roentgen/csvs/orientation_results_valid/results.csv',
              '/workspace/my_auxiliary_persistent/retrain_roentgen/csvs/orientation_results_valid/imgs',
              16)
