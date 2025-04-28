import os
import numpy as np
import torch
import nibabel as nib
import csv
from tqdm import tqdm
from collections import defaultdict
from monai.transforms import Compose, LoadImage, EnsureType, EnsureChannelFirst, ScaleIntensityRange, CropForeground
from lighter_zoo import SegResEncoder
import SimpleITK as sitk
import json

centersrois = {'cyst1': [324, 334, 158],
               'cyst2': [189, 278, 185],
               'hemangioma': [209, 315, 159],
               'metastasis': [111, 271, 140],
               'normal1': [161, 287, 149],
               'normal2': [154, 229, 169]}


patch_size = np.array([64, 64, 32])  # Patch dimensions

# Pre-trained feature extraction model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SegResEncoder.from_pretrained("project-lighter/ct_fm_feature_extractor").to(device)
model.eval()

# Preprocessing pipeline
preprocess = Compose([
    EnsureType(),          # Ensure correct data type
    ScaleIntensityRange(
        a_min=-1024,  
        a_max=2048,  
        b_min=0,  
        b_max=1,  
        clip=True  
    ),
    CropForeground()  
])

# Dictionary to track scanner-dose-IR occurrences
scanner_count = defaultdict(int)

def load_metadata(json_filename):
    """Loads the metadata JSON and returns a list of metadata."""
    with open(json_filename, 'r') as f:
        metadata = json.load(f)  # Parse the JSON file
    return metadata


def extract_patch(image, center, patch_size=(64, 64, 32)):
    """Extracts a 3D patch centered at 'center' with 'patch_size'."""
    # Ensure patch_size is a tuple of three dimensions (depth, height, width)
    assert len(patch_size) == 3
    half_size = [size // 2 for size in patch_size]

    # Create slices for each dimension
    slices = tuple(
        slice(max(0, center[i] - half_size[i]), min(image.shape[i+1], center[i] + half_size[i])) 
        for i in range(3)
    )

    # Extract the patch
    patch = image[:, slices[0], slices[1], slices[2]]

    # If the patch dimensions are smaller than expected (due to the image boundary), 
    # adjust by padding to match the desired size
    pad_depth = patch_size[0] - patch.shape[1]
    pad_height = patch_size[1] - patch.shape[2]
    pad_width = patch_size[2] - patch.shape[3]

    if pad_depth > 0 or pad_height > 0 or pad_width > 0:
        patch = torch.nn.functional.pad(patch, (0, pad_width, 0, pad_height, 0, pad_depth))

    return patch


def run_inference(nifti_image, dicom_folder, output_dir, metadata_json, model, affine, centersrois):
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, "features_ct-fm_full_copia.csv")
    
    # Load metadata from JSON file
    metadata_list = load_metadata(metadata_json)
    
    # Convert Nifti1Image to NumPy array
    image_array = nifti_image.get_fdata(dtype=np.float32)  # Convert to float32

    # Reorder dimensions to (Height, Width, Depth)
    image_array = np.transpose(image_array, (1, 2, 0))  # (343, 512, 512) -> (512, 512, 343)

    image_array = np.expand_dims(image_array, axis=0)  # (512, 512, 343) -> (1, 512, 512, 343)

    # Apply preprocessing
    image_array = preprocess(image_array)

    print(f"Shape after preprocessing: {image_array.shape}")

    # Check if the file already exists
    file_exists = os.path.exists(output_csv)

    with open(output_csv, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            "FileName", "ROI", "SeriesNumber", "SeriesDescription", 
            "ManufacturerModelName", "Manufacturer", "SliceThickness", 
            "StudyDescription", "StudyID", "deepfeatures"
        ])

        # Write header only if the file does not exist
        if not file_exists:
            writer.writeheader()

        for roi_label, center in centersrois.items():
            patch = extract_patch(image_array, np.array(center))

            patches_dir = "/mnt/nas7/data/maria/final_features/patches"
            os.makedirs(patches_dir, exist_ok=True)

            patch_np = patch.squeeze().cpu().numpy()
        
            # Save as NIfTI
            patch_nifti = nib.Nifti1Image(patch_np, affine)
            patch_filename = os.path.join(patches_dir, f"{dicom_folder}_{roi_label}.nii.gz")
            nib.save(patch_nifti, patch_filename)
            print(f"Saved patch: {patch_filename}")

            with torch.no_grad():
                patch = patch.to('cuda')    # Move the tensor to the default GPU
                output = model(patch.unsqueeze(0))[-1]
                feature_vector = torch.nn.functional.adaptive_avg_pool3d(output, 1).squeeze().cpu().numpy()
                formatted_feature_vector = "[" + ", ".join(map(str, feature_vector)) + "]"

            # Find the metadata entry that matches the image and roi_label
            metadata_entry = None
            for item in metadata_list:
                image_filename = os.path.basename(item["image"]).replace(".nii.gz", "")  # Extract filename without extension
                if dicom_folder == image_filename and item["roi_label"] == roi_label:
                    metadata_entry = item
                    break

            print('Processing ROI:', roi_label)

            if metadata_entry is None:
                print(f"Warning: No matching metadata for ROI '{roi_label}' in image '{image_filename}'")
                continue  # Skip this ROI if no matching metadata is found

            # If metadata entry is found, extract the relevant fields
            if metadata_entry:
                metadata = {
                    "FileName": image_filename,
                    "ROI": metadata_entry["roi_label"],  # Use roi_label from the metadata
                    "SeriesNumber": metadata_entry["info"].get("SeriesNumber", "Unknown"),
                    "SeriesDescription": metadata_entry["info"].get("SeriesDescription", "Unknown"),
                    "ManufacturerModelName": metadata_entry["info"].get("ManufacturerModelName", "Unknown"),
                    "Manufacturer": metadata_entry["info"].get("Manufacturer", "Unknown"),
                    "SliceThickness": metadata_entry["info"].get("SliceThickness", "Unknown"),
                    "StudyDescription": metadata_entry["info"].get("StudyDescription", "Unknown"),
                    "StudyID": metadata_entry["info"].get("StudyID", "Unknown")
                }

            else:
                metadata = {
                    "FileName": "Unknown",
                    "ROI": roi_label,
                    "SeriesNumber": "Unknown",
                    "SeriesDescription": "Unknown",
                    "ManufacturerModelName": "Unknown",
                    "Manufacturer": "Unknown",
                    "SliceThickness": "Unknown",
                    "StudyDescription": "Unknown",
                    "StudyID": "Unknown"
                }

            writer.writerow({
                "FileName": metadata["FileName"],
                "ROI": metadata["ROI"],
                "SeriesNumber": metadata["SeriesNumber"],
                "SeriesDescription": metadata["SeriesDescription"],
                "ManufacturerModelName": metadata["ManufacturerModelName"],
                "Manufacturer": metadata["Manufacturer"],
                "SliceThickness": metadata["SliceThickness"],
                "StudyDescription": metadata["StudyDescription"],
                "StudyID": metadata["StudyID"],
                "deepfeatures": formatted_feature_vector
            })


if __name__ == "__main__":
    output_dir = "/mnt/nas7/data/maria/final_features/ct-fm/dicom"
    os.makedirs(output_dir, exist_ok=True)

    json_file = "/mnt/nas7/data/maria/final_features/dicom_metadata.json"

    main_folder = "/mnt/nas4/datasets/ToCurate/QA4IQI/FinalDataset-TCIA-MultiCentric/Upl"
    subfolders = ["A1", "A2", "B1", "B2", "G1", "G2", "C1", "H2", "D1", "E2", "F1", "E1", "H1"]


    for subfolder in subfolders:
        subfolder_path = os.path.join(main_folder, subfolder)

        if not os.path.isdir(subfolder_path):
            print(f"Skipping {subfolder} (not a directory)")
            continue

        print(f"\nProcessing DICOM folder: {subfolder}")

                # Loop through all subfolders inside the main subfolder
        for subfile in sorted(os.listdir(subfolder_path)):
            subfile_path = os.path.join(subfolder_path, subfile)

            reader = sitk.ImageSeriesReader()
            dicom_series = reader.GetGDCMSeriesFileNames(subfile_path)

            print(f"\nProcessing DICOM series: {subfile}")

            if not dicom_series:
                print(f"No DICOM files found in {subfolder}")
                continue

            reader.SetFileNames(dicom_series)
            image = reader.Execute()

            # Convert to NIfTI (without saving)
            nifti_array = sitk.GetArrayFromImage(image)
            #affine = np.eye(4)

            # Extract spacing and origin
            spacing = image.GetSpacing()  # (x, y, z)
            origin = image.GetOrigin()    # (x, y, z)
            direction = np.array(image.GetDirection()).reshape(3, 3)

            # Construct affine matrix
            affine = np.eye(4)
            affine[:3, :3] = direction * spacing  # Scale by spacing
            affine[:3, 3] = origin  # Set origin

            # Convert to NIfTI
            nifti_image = nib.Nifti1Image(nifti_array, affine)

            print(nifti_image.shape)

            print("🔄 Running feature extraction...")
            run_inference(nifti_image, subfile, output_dir, json_file, model, affine, centersrois)
    print("✅ Feature extraction completed!")
    

