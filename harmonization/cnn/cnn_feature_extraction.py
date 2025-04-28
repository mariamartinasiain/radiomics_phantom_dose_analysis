import sys
sys.path.append('/home/maria/radiomics_phantom_copy/')
import tensorflow.compat.v1 as tf

from qa4iqi_extraction.constants import MANUFACTURER_FIELD, MANUFACTURER_MODEL_NAME_FIELD, SERIES_DESCRIPTION_FIELD, SERIES_NUMBER_FIELD, SLICE_THICKNESS_FIELD
from harmonization.swin.extract import custom_collate_fn, load_data, CropOnROId
tf.disable_v2_behavior()
import os
import numpy as np
import csv
from tqdm import tqdm
import nibabel as nib
from monai.transforms import Compose, LoadImage, EnsureType
import torch
import tensorflow as tf2
gpus = tf2.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf2.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

#centersrois = {'cyst1': [324, 334, 158], 'cyst2' : [189, 278, 185], 'hemangioma': [209, 315, 159], 'metastasis': [111, 271, 140], 'normal1': [161, 287, 149], 'normal2': [154, 229, 169]}
#centersrois = {'cyst1': [322, 180, 157], 'cyst2' : [189, 233, 186], 'hemangioma': [212, 193, 159], 
             #  'metastasis': [111, 240, 140], 'normal1': [161, 226, 149], 'normal2': [159, 275, 170]}

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
                "SeriesDescription": row["SeriesDescription"],
                "ManufacturerModelName": row["ManufacturerModelName"],
                "Manufacturer": row["Manufacturer"],
                "SliceThickness": row["SliceThickness"],
                "StudyDescription": row["StudyDescription"],
                "StudyID": row["StudyID"]
            }
    return metadata_dict

def run_inference():
    jsonpath = "/mnt/nas7/data/maria/final_features/expanded_registered_light_dataset_info.json"
    model_dir = '/mnt/nas7/data/maria/final_features/QA4IQI/QA4IQI/'
    model_file = model_dir + 'organs-5c-30fs-acc92-121.meta'

    sess = tf.Session()
    saver = tf.train.import_meta_graph(model_file)
    saver.restore(sess, tf.train.latest_checkpoint(model_dir))
    graph = tf.get_default_graph()
    feature_tensor = graph.get_tensor_by_name('MaxPool3D_1:0')
    x = graph.get_tensor_by_name('x_start:0') 
    keepProb = graph.get_tensor_by_name('keepProb:0')  

    device_id = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    torch.cuda.set_device(device_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    transforms = Compose([
        LoadImage(ensure_channel_first=True),
        EnsureType(),
    ])

    nifti_dir = "/mnt/nas7/data/reza/registered_dataset_all_doses_pad_updated_NotRegistered/"
    datafiles = sorted([f for f in os.listdir(nifti_dir) if f.endswith(".nii.gz")])

    output_dir = "/mnt/nas7/data/maria/final_features/"
    output_file = os.path.join(output_dir, "cnn_features_not_registered.csv")

    metadata_csv = "/mnt/nas7/data/maria/final_features/ct-fm/dicom_metadata/dicom_metadata.csv"
    metadata_dict = load_metadata(metadata_csv)

    # Step 1: Read already processed files + ROIs
    processed = set()
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                processed.add((row["FileName"], row["ROI"]))

    # Step 2: Open CSV in append mode
    with open(output_file, "a", newline="") as csvfile:
        fieldnames = ["FileName", "SeriesNumber", "deepfeatures", "ROI", "SeriesDescription", "ManufacturerModelName", "Manufacturer", "SliceThickness", "StudyDescription", "StudyID"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header only if file was empty
        if os.stat(output_file).st_size == 0:
            writer.writeheader()

        for file in tqdm(datafiles, desc="Processing CT scans"):
            file_name = file.replace(".nii.gz", "")

            # Check which ROIs still need to be processed for this file
            missing_rois = [roi_label for roi_label in centersrois if (file, roi_label) not in processed]

            if not missing_rois:
                print(file, "already fully processed, skipping loading")
                continue

            print('Processing file', file_name)
            # Only now load the NIfTI file!
            file_path = os.path.join(nifti_dir, file)
            image = transforms(file_path)
            image = image.squeeze(0)  
            image = image.permute(1, 0, 2) 
            image = torch.flip(image, dims=[0])  
            image = image.unsqueeze(0)

            if file_name in metadata_dict:
                metadata = metadata_dict[file_name]
                series_description = metadata["SeriesDescription"]
                manufacturer_model_name = metadata["ManufacturerModelName"]
                manufacturer = metadata["Manufacturer"]
                slicethickness = metadata["SliceThickness"]
                study_description = metadata["StudyDescription"]
                study_id = metadata["StudyID"]
            else:
                series_description = "Unknown"
                manufacturer_model_name = "Unknown"
                manufacturer = "Unknown"
                slicethickness = "Unknown"
                study_description = "Unknown"
                study_id = "Unknown"

            for roi_label in missing_rois:
                patch_center = np.array(centersrois[roi_label])
                patch = extract_patch(image, patch_center)

                flattened_image = patch.flatten()
                flattened_image = np.array(flattened_image)
                flattened_image = flattened_image.reshape(1, -1)

                features = sess.run(feature_tensor, feed_dict={x: flattened_image, keepProb: 1.0})
                latentrep = sess.run(tf.reshape(features, [-1])).tolist()

                writer.writerow({
                    "FileName": file,
                    "SeriesNumber": "Unknown",
                    "deepfeatures": latentrep,
                    "ROI": roi_label,
                    "SeriesDescription": series_description,
                    "ManufacturerModelName": manufacturer_model_name,
                    "Manufacturer": manufacturer,
                    "SliceThickness": slicethickness,
                    "StudyDescription": study_description,
                    "StudyID": study_id
                })
                processed.add((file, roi_label))

    print("Done!")


if __name__ == "__main__":
    run_inference()
   
'''

import pandas as pd  

# Cargar el archivo combinado  
file_path = "/mnt/nas7/data/maria/final_features/cnn_features_complete_updated.csv"
df = pd.read_csv(file_path)  

# Contar valores únicos en SeriesNumber  
unique_series_numbers = df["SeriesNumber"].nunique()  
print("Número de SeriesNumber únicos:", unique_series_numbers)  

# Extraer la dosis de SeriesDescription usando regex  
df["Dosis"] = df["SeriesDescription"].str.extract(r'(\d+mGy)')  

# Contar apariciones de cada dosis  
dosis_values = ["1mGy", "3mGy", "6mGy", "10mGy", "14mGy"]  
dosis_counts = df["Dosis"].value_counts().reindex(dosis_values, fill_value=0)  

print("\nFrecuencia de cada dosis en SeriesDescription:")  
print(dosis_counts)

import pandas as pd  

# File paths  
features_file = "/mnt/nas7/data/maria/final_features/cnn_features_complete_updated.csv"  
metadata_file = "/mnt/nas7/data/maria/final_features/ct-fm/dicom_metadata/dicom_metadata.csv"  

# Load data  
df_features = pd.read_csv(features_file)  
df_metadata = pd.read_csv(metadata_file)  

# Extract unique SeriesNumber values  
features_series_numbers = set(df_features["StudyID"].dropna().unique())  
metadata_series_numbers = set(df_metadata["StudyID"].dropna().unique())  

# Find missing SeriesNumber values  
missing_series_numbers = metadata_series_numbers - features_series_numbers  

print(len(metadata_series_numbers))

print("Missing SeriesNumber values:", missing_series_numbers)  
print(f"Total missing: {len(missing_series_numbers)}")


'''