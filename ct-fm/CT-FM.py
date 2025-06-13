import os
import numpy as np
import torch
import nibabel as nib
import csv
from tqdm import tqdm
from collections import defaultdict
from monai.transforms import Compose, LoadImage, EnsureType, Orientation, ScaleIntensityRange, CropForeground
from lighter_zoo import SegResEncoder


# ROI centers (fixed after registration)
centersrois = {
    'cyst1': [260, 214, 145],  
    'cyst2': [125, 158, 172], 
    'hemangioma': [145, 195, 146],  
    'metastasis': [47, 151, 127],  
    'normal1': [97, 167, 136],  
    'normal2': [90, 109, 156] 
}

patch_size = np.array([64, 64, 32])  # Patch dimensions

# Pre-trained feature extraction model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SegResEncoder.from_pretrained("project-lighter/ct_fm_feature_extractor").to(device)
model.eval()

# Preprocessing pipeline
preprocess = Compose([
    LoadImage(ensure_channel_first=True),
    EnsureType(),                         
    ScaleIntensityRange(
        a_min=-1024,    
        a_max=2048,     
        b_min=0,        
        b_max=1,        
        clip=True       
    ),
    CropForeground()   
])

scanner_count = defaultdict(int)

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

def load_metadata(csv_filename):
    """Loads the metadata CSV and returns a dictionary of folder names to metadata."""
    metadata_dict = {}
    with open(csv_filename, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            folder_name = row["Folder"]
            metadata_dict[folder_name] = {
                "SeriesDescription": row["SeriesDescription"],
                "ManufacturerModelName": row["ManufacturerModelName"],
                "Manufacturer": row["Manufacturer"],
                "SliceThickness": row["SliceThickness"]
            }
    return metadata_dict

def run_inference(nifti_dir, output_dir, metadata_dict):
    nifti_files = [f for f in os.listdir(nifti_dir) if f.endswith(".nii.gz")]
    os.makedirs(output_dir, exist_ok=True)
    patches_dir = os.path.join(output_dir, "patches")
    os.makedirs(patches_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, "features_ct-fm_full.csv")
    
    with open(output_csv, "w", newline="") as csvfile:
        fieldnames = ["FileName", "ROI", "deepfeatures", "SeriesDescription", "ManufacturerModelName", "Manufacturer", "SliceThickness"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for file in tqdm(nifti_files, desc="Processing CT scans"):
            file_path = os.path.join(nifti_dir, file)
            image = preprocess(file_path)
            
            file_name = file.replace(".nii.gz", "")

            if file_name in metadata_dict:
                metadata = metadata_dict[file_name]
                series_description = metadata["SeriesDescription"]
                manufacturer_model_name = metadata["ManufacturerModelName"]
                manufacturer = metadata["Manufacturer"]
                slicethickness = metadata["SliceThickness"]
            else:
                series_description = "Unknown"
                manufacturer_model_name = "Unknown"
                manufacturer = "Unknown"
                slicethickness = "Unknown"

            for roi_label, center in centersrois.items():
                patch_center = np.array(center)  # Original ROI center
                # Adjust ROI center based on the crop region
                patch = extract_patch(image, patch_center)

                with torch.no_grad():
                    patch = patch.to('cuda')    
                    output = model(patch.unsqueeze(0))[-1]
                    feature_vector = torch.nn.functional.adaptive_avg_pool3d(output, 1).squeeze()
                    feature_vector = feature_vector.cpu().numpy() 
                    formatted_feature_vector = "[" + ", ".join([str(x) for x in feature_vector]) + "]"
                
                writer.writerow({
                    "FileName": file,
                    "ROI": roi_label,
                    "deepfeatures": formatted_feature_vector,
                    "SeriesDescription": series_description,
                    "ManufacturerModelName": manufacturer_model_name,
                    "Manufacturer": manufacturer,
                    "SliceThickness": slicethickness
                })

    print("✅ Feature extraction and patch saving completed!")

if __name__ == "__main__":
    nifti_dir = "/mnt/nas7/data/reza/registered_dataset_all_doses"
    output_dir = "/mnt/nas7/data/maria/final_features/ct-fm/prueba"
    os.makedirs(output_dir, exist_ok=True)
    metadata_csv = "/mnt/nas7/data/maria/final_features/ct-fm/dicom_metadata/dicom_metadata.csv"
    
    metadata_dict = load_metadata(metadata_csv)
    run_inference(nifti_dir, output_dir, metadata_dict)

