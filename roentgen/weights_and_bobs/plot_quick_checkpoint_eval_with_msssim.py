import re
import matplotlib.pyplot as plt
import numpy as np


results_file_path = "/workspace/my_auxiliary_persistent/retrain_roentgen/eval_checkpoints/quick_eval_results_with_msssim.txt"
plot_save_path = "/workspace/my_auxiliary_persistent/retrain_roentgen/eval_checkpoints/fid_scores_plot_with_msssim.png"


checkpoints = []
fid_inception_values = []
fid_clip_values = []
fid_xrv_values = []
ms_ssim_values = []  


checkpoint_pattern = r"Evaluating checkpoint:\s*(.+)"
fid_inception_pattern = r"FID InceptionV3:\s*([\d.]+)"
fid_clip_pattern = r"FID CLIP-ViT-B/32:\s*([\d.]+)"
fid_xrv_pattern = r"FID DenseNet-121 \(XRV\):\s*([\d.]+)"
ms_ssim_pattern = r"Mean MS-SSIM:\s*([\d.]+)"  


with open(results_file_path, 'r') as file:
    current_checkpoint = None
    inception, clip, xrv, msssim = None, None, None, None  # temp variables to store each metric
    
    for line in file:
        line = line.strip()

        # match and capture each checkpoint path
        checkpoint_match = re.search(checkpoint_pattern, line)
        if checkpoint_match:
            # save values if they exist and reset for the next checkpoint
            if current_checkpoint and inception and clip and xrv and msssim:
                checkpoints.append(current_checkpoint[75:])  # Shortened name
                fid_inception_values.append(float(inception))
                fid_clip_values.append(float(clip))
                fid_xrv_values.append(float(xrv))
                ms_ssim_values.append(float(msssim))  # Add MS-SSIM value
                print(f"Saved data for {current_checkpoint}")

            # reset values for the new checkpoint
            current_checkpoint = checkpoint_match.group(1)
            inception, clip, xrv, msssim = None, None, None, None
            print(f"Found checkpoint: {current_checkpoint}")

        # Capture FID and MS-SSIM values if the checkpoint path has been set
        if current_checkpoint:
            inception_match = re.search(fid_inception_pattern, line)
            clip_match = re.search(fid_clip_pattern, line)
            xrv_match = re.search(fid_xrv_pattern, line)
            msssim_match = re.search(ms_ssim_pattern, line)  # check for MS-SSIM value

            # xapture each FID score if matched
            if inception_match:
                inception = inception_match.group(1)
                print(f"Found FID InceptionV3: {inception}")
            if clip_match:
                clip = clip_match.group(1)
                print(f"Found FID CLIP-ViT-B/32: {clip}")
            if xrv_match:
                xrv = xrv_match.group(1)
                print(f"Found FID DenseNet-121 (XRV): {xrv}")
            if msssim_match:
                msssim = msssim_match.group(1)
                print(f"Found Mean MS-SSIM: {msssim}")

    # last save after loop if any data is pending
    if current_checkpoint and inception and clip and xrv and msssim:
        checkpoints.append(current_checkpoint.split('/')[-1].replace("checkpoint-", ""))
        fid_inception_values.append(float(inception))
        fid_clip_values.append(float(clip))
        fid_xrv_values.append(float(xrv))
        ms_ssim_values.append(float(msssim))  # Add final MS-SSIM value
        print(f"Final data saved for {current_checkpoint}")

print("CHECKPOINTS")
print(checkpoints)

# verify data lists match in length before plotting
if len(checkpoints) == len(fid_inception_values) == len(fid_clip_values) == len(fid_xrv_values) == len(ms_ssim_values):
    x = np.arange(len(checkpoints))  # label locations
    width = 0.2  # adjusted bar width for four bars

    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot each FID score category and msssssim in a bar
    bars1 = ax.bar(x - 1.5 * width, fid_inception_values, width, label='FID InceptionV3', color='skyblue')
    bars2 = ax.bar(x - 0.5 * width, fid_clip_values, width, label='FID CLIP-ViT-B/32', color='salmon')
    bars3 = ax.bar(x + 0.5 * width, fid_xrv_values, width, label='FID DenseNet-121 (XRV)', color='lightgreen')
    bars4 = ax.bar(x + 1.5 * width, ms_ssim_values, width, label='Mean MS-SSIM', color='mediumpurple')

 
    ax.set_xlabel('Checkpoint')
    ax.set_ylabel('Score')
    ax.set_title('FID Scores and Mean MS-SSIM by Checkpoint')
    ax.set_xticks(x)
    ax.set_xticklabels(checkpoints, rotation=45, ha="right", fontsize=10)  
    ax.legend()

    # Add value labels for verification
    for bar in bars1 + bars2 + bars3 + bars4:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', ha='center', va='bottom')

   
    plt.tight_layout(pad=2)
    plt.savefig(plot_save_path)
    print(f"Plot saved successfully at {plot_save_path}")
else:
    print("Data lengths are mismatched; please verify the input file.")

