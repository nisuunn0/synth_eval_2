import re
import matplotlib.pyplot as plt
import numpy as np


results_file_path = "/workspace/my_auxiliary_persistent/retrain_roentgen/eval_checkpoints/quick_eval_results.txt"
plot_save_path = "/workspace/my_auxiliary_persistent/retrain_roentgen/eval_checkpoints/fid_scores_plot.png"


checkpoints = []
fid_inception_values = []
fid_clip_values = []
fid_xrv_values = []

# regex for matching lines
checkpoint_pattern = r"Evaluating checkpoint:\s*(.+)"
fid_inception_pattern = r"FID InceptionV3:\s*([\d.]+)"
fid_clip_pattern = r"FID CLIP-ViT-B/32:\s*([\d.]+)"
fid_xrv_pattern = r"FID DenseNet-121 \(XRV\):\s*([\d.]+)"

# get checkpoint data
with open(results_file_path, 'r') as file:
    current_checkpoint = None
    inception, clip, xrv = None, None, None 
    
    for line in file:
        line = line.strip()

        # 
        checkpoint_match = re.search(checkpoint_pattern, line)
        if checkpoint_match:
           
            if current_checkpoint and inception and clip and xrv:
                #checkpoints.append(current_checkpoint.split('/')[-1].replace("checkpoint-", "")) 
                checkpoints.append(current_checkpoint[75:])
                fid_inception_values.append(float(inception))
                fid_clip_values.append(float(clip))
                fid_xrv_values.append(float(xrv))
                print(f"Saved data for {current_checkpoint}")

            
            current_checkpoint = checkpoint_match.group(1)
            inception, clip, xrv = None, None, None
            print(f"Found checkpoint: {current_checkpoint}")

        # xapture FID values if the checkpoint path has been set
        if current_checkpoint:
            inception_match = re.search(fid_inception_pattern, line)
            clip_match = re.search(fid_clip_pattern, line)
            xrv_match = re.search(fid_xrv_pattern, line)

            # get each FID score if matched
            if inception_match:
                inception = inception_match.group(1)
                print(f"Found FID InceptionV3: {inception}")
            if clip_match:
                clip = clip_match.group(1)
                print(f"Found FID CLIP-ViT-B/32: {clip}")
            if xrv_match:
                xrv = xrv_match.group(1)
                print(f"Found FID DenseNet-121 (XRV): {xrv}")

    # Final save after loop if any data is pending
    if current_checkpoint and inception and clip and xrv:
        checkpoints.append(current_checkpoint.split('/')[-1].replace("checkpoint-", ""))
        fid_inception_values.append(float(inception))
        fid_clip_values.append(float(clip))
        fid_xrv_values.append(float(xrv))
        print(f"Final data saved for {current_checkpoint}")

print("CHECKPOINTS")
print(checkpoints)

# Verify data lists match in length before plotting
if len(checkpoints) == len(fid_inception_values) == len(fid_clip_values) == len(fid_xrv_values):
    x = np.arange(len(checkpoints))  # label locations
    width = 0.25  # bar width

    fig, ax = plt.subplots(figsize=(12, 8))

    #pPlot each FID score category in a bar
    bars1 = ax.bar(x - width, fid_inception_values, width, label='FID InceptionV3', color='skyblue')
    bars2 = ax.bar(x, fid_clip_values, width, label='FID CLIP-ViT-B/32', color='salmon')
    bars3 = ax.bar(x + width, fid_xrv_values, width, label='FID DenseNet-121 (XRV)', color='lightgreen')

    # vonfigure labels and title
    ax.set_xlabel('Checkpoint')
    ax.set_ylabel('FID Score')
    ax.set_title('FID Scores by Checkpoint')
    ax.set_xticks(x)
    ax.set_xticklabels(checkpoints, rotation=45, ha="right", fontsize=10)  # Rotate and align for readability
    ax.legend()

    # add value labels for verification
    for bar in bars1 + bars2 + bars3:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', ha='center', va='bottom')

    # adjust layout and padding to ensure labels fit
    plt.tight_layout(pad=2)
    plt.savefig(plot_save_path)
    print(f"Plot saved successfully at {plot_save_path}")
else:
    print("Data lengths are mismatched; please verify the input file.")

