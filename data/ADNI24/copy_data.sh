#!/bin/bash

# Define source and destination paths
SRC_DIR="/media/smbshare/Saranathan_share/ADNI24/images"
IMG_DST="/home/srs-9/Projects/saranathan/kira_share/deepthomas/data/ADNI24/imagesTs"
LBL_DST="/home/srs-9/Projects/saranathan/kira_share/deepthomas/data/ADNI24/labelsTs"

# Create destination directories if they don't exist
mkdir -p "$IMG_DST" "$LBL_DST"

# Loop through each subject directory
for subj_path in "$SRC_DIR"/*_S_*; do
    # Ensure it's actually a directory
    [ -d "$subj_path" ] || continue
    
    # Extract the directory name (e.g., 002_S_4213)
    subj_id=$(basename "$subj_path")
    
    # 1. Copy and rename the image file
    src_img="$subj_path/left/crop_${subj_id}.nii.gz"
    if [ -f "$src_img" ]; then
        cp "$src_img" "$IMG_DST/crop_${subj_id}.nii.gz"
    else
        echo "Warning: Image missing for $subj_id"
    fi

    # 2. Copy and rename the label file
    src_lbl="$subj_path/sthomas_LR_labels.nii.gz"
    if [ -f "$src_lbl" ]; then
        cp "$src_lbl" "$LBL_DST/crop_${subj_id}_sthomas_LR_labels.nii.gz"
    else
        echo "Warning: Label missing for $subj_id"
    fi
done

echo "Copy and rename process completed successfully!"
