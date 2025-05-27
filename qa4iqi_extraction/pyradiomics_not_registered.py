from nifti_masks_generation import generate_nifti_masks
from extract_pyradiomics_features import extract_features
import os
import nibabel as nib
import pandas as pd
import glob

# Define base paths
nifti_dataset = "/mnt/nas7/data/reza/registered_dataset_all_doses_pad_updated_NotRegistered/"
dicom_data_root = "/mnt/nas4/datasets/ToCurate/QA4IQI/FinalDataset-TCIA-MultiCentric/Upl"
dicom_subfolders = ["A1", "A2", "B1", "B2", "G1", "G2", "C1", "H2", "D1", "E2", "F1", "E1", "H1"]

output_path = "/mnt/nas7/data/maria/final_features/pyradiomics_extraction/not_registered"
os.makedirs(output_path, exist_ok=True)

output_path_rois = "/mnt/nas7/data/maria/final_features/pyradiomics_extraction/rois/not_registered"
os.makedirs(output_path_rois, exist_ok=True)

json_file = "/mnt/nas7/data/maria/final_features/dicom_metadata.json"
output_dir = "/mnt/nas7/data/maria/final_features/"
output_file = os.path.join(output_dir, "pyradiomics_features_not_registered.csv")

# Track already processed files
if os.path.exists(output_file):
    existing_df = pd.read_csv(output_file, usecols=["FileName"])
    processed_files = set(existing_df["FileName"].astype(str))
else:
    processed_files = set()

# Iterate over NIfTI images
for image_file in sorted(os.listdir(nifti_dataset)):
    if image_file.endswith(".nii") or image_file.endswith(".nii.gz"):
        file_name = image_file.replace(".nii.gz", "").replace(".nii", "")
        
        if file_name in processed_files:
            print(f"Skipping {image_file} (already processed)")
            continue

        image_path = os.path.join(nifti_dataset, image_file)
        reference_nifti = nib.load(image_path)

        # Try to locate the mask in each subfolder
        dicom_mask_path = None
        matched_full_subfolder = None
        for subfolder in dicom_subfolders:
            search_pattern = os.path.join(dicom_data_root, subfolder, "*", "mask", file_name + ".dcm")
            matching_paths = glob.glob(search_pattern)
            if matching_paths:
                dicom_mask_path = matching_paths[0]
                print('dicom_mask_path:', dicom_mask_path)

                # Get full subfolder path like: A1/A1_174008_... instead of just 'A1'
                matched_full_subfolder = os.path.relpath(os.path.dirname(os.path.dirname(dicom_mask_path)), dicom_data_root)
                break

        if dicom_mask_path is None:
            print(f"No mask found for {file_name}, skipping.")
            continue

        try:
            roi_masks_paths = generate_nifti_masks(
                dicom_mask_path,
                reference_nifti,
                output_path,
                output_path_rois,
                dicom_data_root,
                [matched_full_subfolder]  # <- full path relative to dicom_data_root
            )
        except Exception as e:
            print(f"Error generating mask for {file_name}: {e}")
            continue

        features_df = extract_features(image_path, roi_masks_paths, json_file)

        if not features_df.empty:
            file_exists = os.path.exists(output_file)
            features_df.to_csv(output_file, mode='a', header=not file_exists, index=False)
            print(f"Saved features from {image_file} to {output_file}")

print("Feature extraction complete!")
