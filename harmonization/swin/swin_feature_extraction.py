import os
import numpy as np
import torch
import nibabel as nib
import csv
from tqdm import tqdm
from monai.transforms import Compose, LoadImage, EnsureType, ScaleIntensityRange, CropForeground
from lighter_zoo import SegResEncoder
import torch.nn.functional as F
from utils import load_data, get_model, get_model_oscar


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


checkpoint_path = "/home/reza/radiomics_phantom/checkpoints/model_swinvit.pt"
model = get_model(model_path=checkpoint_path)

model.to(device)
model.eval()
#model.swinViT.eval()

# Preprocessing pipeline
preprocess = Compose([
    LoadImage(ensure_channel_first=True),
    EnsureType(),
])

centersrois = {'cyst1': [180, 322, 157], 'cyst2' : [233, 189, 186], 'hemangioma': [193, 212, 159], 'metastasis': [240, 111, 140], 'normal1': [226, 161, 149], 'normal2': [275, 159, 170]}


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
                "SeriesNumber": row["SeriesNumber"],
                "SeriesDescription": row["SeriesDescription"],
                "ManufacturerModelName": row["ManufacturerModelName"],
                "Manufacturer": row["Manufacturer"],
                "SliceThickness": row["SliceThickness"],
                "StudyDescription": row["StudyDescription"],
                "StudyID": row["StudyID"]
            }
    return metadata_dict


def run_inference():

    nifti_dir = "/mnt/nas7/data/reza/registered_dataset_all_doses_pad_updated_NotRegistered/"
    datafiles = sorted([f for f in os.listdir(nifti_dir) if f.endswith(".nii.gz")])

    output_dir = "/mnt/nas7/data/maria/final_features/"
    output_file = os.path.join(output_dir, "swinunetr_features_not_registered.csv")

    metadata_csv = "/mnt/nas7/data/maria/final_features/ct-fm/dicom_metadata/dicom_metadata.csv"
    metadata_dict = load_metadata(metadata_csv)

    patches_dir = os.path.join(output_dir, "patches")


    # Open CSV file once
    with open(output_file, "w", newline="") as csvfile:
        fieldnames = ["FileName", "SeriesNumber", "deepfeatures", "ROI", "SeriesDescription", "ManufacturerModelName", "Manufacturer", "SliceThickness", "StudyDescription", "StudyID"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for file in tqdm(datafiles, desc="Processing CT scans"):
            print(f"Processing {file}")
            file_name = file.replace(".nii.gz", "")

            file_path = os.path.join(nifti_dir, file)
            image = preprocess(file_path)
            image = image.squeeze(0)  
            image = image.permute(1, 0, 2) 
            image = torch.flip(image, dims=[0])  
            image = image.unsqueeze(0)

            if file_name in metadata_dict:
                metadata = metadata_dict[file_name]
                series_number = metadata["SeriesNumber"]
                series_description = metadata["SeriesDescription"]
                manufacturer_model_name = metadata["ManufacturerModelName"]
                manufacturer = metadata["Manufacturer"]
                slicethickness = metadata["SliceThickness"]
                study_description = metadata["StudyDescription"]
                study_id = metadata["StudyID"]
            else:
                series_number = "Unknown"
                series_description = "Unknown"
                manufacturer_model_name = "Unknown"
                manufacturer = "Unknown"
                slicethickness = "Unknown"
                study_description = "Unknown"
                study_id = "Unknown"

            for roi_label, center in centersrois.items():
                patch_center = np.array(centersrois[roi_label])
                patch = extract_patch(image, patch_center)
                patch = patch.unsqueeze(1)

                # Save the patch as a NIfTI file
                #patch_np = masked_image_cropped.squeeze(0).cpu().numpy()  # remove channel dimension, move to CPU
                #patch_affine = np.eye(4)  # or get it from the original image if needed

                #patch_nii = nib.Nifti1Image(patch_np, affine=patch_affine)
                #patch_filename = os.path.join(patches_dir, f"{file}_ROI_{name}.nii.gz")
                #nib.save(patch_nii, patch_filename)

                # Feature extraction
                with torch.no_grad():
                    patch = patch.to('cuda')
                    #val_outputs = model.swinViT(patch.float())
                    val_outputs = model.swinViT(patch)
                    latentrep = val_outputs[4]  # Shape: [1, 768, 2, 2, 1]

                latentrep = latentrep.mean(dim=[2, 3, 4])  # Average → shape: [1, 768]
                feature_vector = latentrep.squeeze(0).cpu().numpy()  # Final shape: [768]
                formatted_feature_vector = "[" + ", ".join(map(str, feature_vector)) + "]"

                writer.writerow({
                    "FileName": file,
                    "SeriesNumber": series_number,
                    "deepfeatures": formatted_feature_vector,
                    "ROI": roi_label,
                    "SeriesDescription": series_description,
                    "ManufacturerModelName": manufacturer_model_name,
                    "Manufacturer": manufacturer,
                    "SliceThickness": slicethickness,
                    "StudyDescription": study_description,
                    "StudyID": study_id
                })
                

    print("Done!")

# Paths
if __name__ == "__main__":
    run_inference()