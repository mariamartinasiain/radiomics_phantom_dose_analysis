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
    # Adjusted coordinates (after crop)
    'cyst1': [260, 214, 145],  # (324 - 13, 334 - 120, 158 - 64)
    'cyst2': [125, 158, 172],  # (189 - 13, 278 - 120, 185 - 64)
    'hemangioma': [145, 195, 146],  # (209 - 13, 315 - 120, 159 - 64)
    'metastasis': [47, 151, 127],  # (111 - 13, 271 - 120, 140 - 64)
    'normal1': [97, 167, 136],  # (161 - 13, 287 - 120, 149 - 64)
    'normal2': [90, 109, 156]  # (154 - 13, 229 - 120, 169 - 64)
}

patch_size = np.array([64, 64, 32])  # Patch dimensions

# Dictionary to map ManufacturerModelName to Manufacturer
manufacturer_map = {
    "SOMATOM X cite": "Siemens Healthineers",
    "iCT 256": "Philips",
    "Aquilion Prime SP": "TOSHIBA",
    "SOMATOM Definition Flash": "SIEMENS",
    "SOMATOM Definition Edge": "SIEMENS",
    "Revolution EVO": "GE MEDICAL SYSTEMS",
    "Revolution Apex": "GE MEDICAL SYSTEMS",
    "BrightSpeed S": "GE MEDICAL SYSTEMS",
    "Brilliance 64": "Philips",
    "SOMATOM Edge Plus": "SIEMENS",
    "Aquilion": "TOSHIBA"
}

# Pre-trained feature extraction model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SegResEncoder.from_pretrained("project-lighter/ct_fm_feature_extractor").to(device)
model.eval()

# Preprocessing pipeline
preprocess = Compose([
    LoadImage(ensure_channel_first=True),  # Load image and ensure channel dimension
    EnsureType(),                         # Ensure correct data type
    #Orientation(axcodes="SPL"),           # Standardize orientation
    # Scale intensity to [0,1] range, clipping outliers
    ScaleIntensityRange(
        a_min=-1024,    # Min HU value
        a_max=2048,     # Max HU value
        b_min=0,        # Target min
        b_max=1,        # Target max
        clip=True       # Clip values outside range
    ),
    CropForeground()    # Remove background to reduce computation
])

# Dictionary to track scanner-dose-IR occurrences
scanner_count = defaultdict(int)

def extract_patch(image, center, patch_size=(64, 64, 32)):
    """Extracts a 3D patch centered at 'center' with 'patch_size'."""
    # Ensure patch_size is a tuple of three dimensions (depth, height, width)
    assert len(patch_size) == 3
    
    # Calculate half size of the patch in each dimension
    half_size = [size // 2 for size in patch_size]

    # Create slices for each dimension
    slices = tuple(
        slice(max(0, center[i] - half_size[i]), min(image.shape[i+1], center[i] + half_size[i])) 
        for i in range(3)
    )

    # Extract the patch using the calculated slices
    patch = image[:, slices[0], slices[1], slices[2]]

    # If the patch dimensions are smaller than expected (due to the image boundary), 
    # adjust by padding to match the desired size
    pad_depth = patch_size[0] - patch.shape[1]
    pad_height = patch_size[1] - patch.shape[2]
    pad_width = patch_size[2] - patch.shape[3]

    if pad_depth > 0 or pad_height > 0 or pad_width > 0:
        patch = torch.nn.functional.pad(patch, (0, pad_width, 0, pad_height, 0, pad_depth))

    return patch

def extract_metadata(filename):
    """Extracts SeriesDescription and ManufacturerModelName from filename."""
    parts = filename.split("_")
    scanner_id = parts[0]  # A1
    manufacturer_model_name = " ".join(parts[3:6])  # SOMATOM_Definition_Edge
    dose_info = "_".join(parts[-4:-2])  # Harmonized_14mGy_IR
    key = f"{scanner_id}_{dose_info}"
    scanner_count[key] += 1
    series_description = f"{scanner_id}_{dose_info} - #{scanner_count[key]}"

    return {"SeriesDescription": series_description, "ManufacturerModelName": manufacturer_model_name}

def run_inference(nifti_dir, output_dir):
    nifti_files = [f for f in os.listdir(nifti_dir) if f.endswith(".nii.gz")]
    os.makedirs(output_dir, exist_ok=True)
    patches_dir = os.path.join(output_dir, "patches")
    os.makedirs(patches_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, "features_ct_fm_full.csv")
    
    with open(output_csv, "w", newline="") as csvfile:
        fieldnames = ["FileName", "ROI", "deepfeatures", "SeriesDescription", "ManufacturerModelName", "Manufacturer"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for file in tqdm(nifti_files, desc="Processing CT scans"):
            file_path = os.path.join(nifti_dir, file)
            image = preprocess(file_path)
       
            metadata = extract_metadata(file)
            series_description = metadata["SeriesDescription"]
            manufacturer_model_name = metadata["ManufacturerModelName"]
            # Get manufacturer from the model name (using the mapping)
            manufacturer = manufacturer_map.get(manufacturer_model_name, "Unknown")  # Default to "Unknown" if not found

            for roi_label, center in centersrois.items():
                patch_center = np.array(center)  # Original ROI center
                # Adjust ROI center based on the crop region
                patch = extract_patch(image, patch_center)

                with torch.no_grad():

                    patch = patch.to('cuda')    # Move the tensor to the default GPU

                    output = model(patch.unsqueeze(0))[-1]
                    feature_vector = torch.nn.functional.adaptive_avg_pool3d(output, 1).squeeze()
                    feature_vector = feature_vector.cpu().numpy()  # Convert tensor to numpy array
                    #formatted_feature_vector = "[" + ", ".join([f"{x:.4f}" for x in feature_vector]) + "]"
                    formatted_feature_vector = "[" + ", ".join([str(x) for x in feature_vector]) + "]"
                
                writer.writerow({
                    "FileName": file,
                    "ROI": roi_label,
                    "deepfeatures": formatted_feature_vector,
                    "SeriesDescription": series_description,
                    "ManufacturerModelName": manufacturer_model_name,
                    "Manufacturer": manufacturer
                })

    print("✅ Feature extraction and patch saving completed!")

if __name__ == "__main__":
    nifti_dir = "/mnt/nas7/data/reza/registered_dataset_all_doses"
    output_dir = "/mnt/nas7/data/maria/final_features/ct-fm"
    run_inference(nifti_dir, output_dir)
