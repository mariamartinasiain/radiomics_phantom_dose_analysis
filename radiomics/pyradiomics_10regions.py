from nifti_masks_generation import generate_nifti_masks, generate_nifti_masks_10regions
from extract_pyradiomics_features import extract_features, extract_features_10regions
import os
import nibabel as nib
import pandas as pd
import json

# Define paths
dicom_mask = '/mnt/nas4/datasets/ToCurate/QA4IQI/FinalDataset-TCIA-MultiCentric/Upl/A1/' \
'A1_174008_691000_SOMATOM_Definition_Edge_ID23_Harmonized_10mGy_IR_NrFiles_343/mask/A1_174008_691000_SOMATOM_Definition_Edge_ID23_Harmonized_10mGy_IR_NrFiles_343.dcm'  # DICOM file containing the segmentation mask

reference_nifti = '/mnt/nas7/data/reza/registered_dataset_all_doses_pad_updated_NotRegistered/A1_174008_691000_SOMATOM_Definition_Edge_ID23_Harmonized_10mGy_IR_NrFiles_343.nii.gz'

output_path = "/mnt/nas7/data/maria/final_features/pyradiomics_extraction/10_regions"
os.makedirs(output_path, exist_ok=True)

output_path_rois = "/mnt/nas7/data/maria/final_features/pyradiomics_extraction/rois/10_regions"
os.makedirs(output_path_rois, exist_ok=True)

centersrois = "/home/maria/radiomics_phantom_copy/qa4iqi_extraction/centersrois_10.json"
path_size = (32, 32, 16)  # Patch size (depth, height, width)

# Run the function to generate the ROI nifti masks
#generate_nifti_masks_10regions(reference_nifti, centersrois, output_path_rois, patch_size=(32, 32, 16))

# Extract Pyradiomics features
nifti_dataset = "/mnt/nas7/data/reza/registered_dataset_all_doses_pad_updated_NotRegistered/"

all_features_df = []

json_file = "/mnt/nas7/data/maria/final_features/dicom_metadata.json"


output_file = os.path.join(output_path, "pyradiomics_features_10regions.csv")

if os.path.exists(output_file):
    existing_df = pd.read_csv(output_file, usecols=["FileName"])  # Read only the "FileName" column
    processed_files = set(existing_df["FileName"].astype(str))  # Convert to a set for quick lookup
else:
    processed_files = set()  # If no file exists, start with an empty set

# Iterate over images
for image_file in sorted(os.listdir(nifti_dataset)):
    if ("10mGy" not in image_file):  # Only process images with '10mGy' in the name
        continue

    if image_file.endswith(".nii") or image_file.endswith(".nii.gz"):  # Check for medical image formats
        file_name = image_file.replace(".nii.gz", "").replace(".nii", "")  # Normalize filename
        
        if file_name in processed_files:
            print(f"Skipping {image_file} (already processed)")
            continue  # Skip feature extraction for this file

        image_path = os.path.join(nifti_dataset, image_file)
        
        # Extract features
        features_df = extract_features_10regions(image_path, output_path_rois, json_file)

        # Append to CSV after each image processing
        if not features_df.empty:
            file_exists = os.path.exists(output_file)
            features_df.to_csv(output_file, mode='a', header=not file_exists, index=False)

            print(f"Saved features from {image_file} to {output_file}")

print("Feature extraction complete!")
