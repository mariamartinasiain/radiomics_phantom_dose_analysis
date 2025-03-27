import sys
sys.path.append('/home/maria/radiomics_phantom_copy/')
import tensorflow.compat.v1 as tf

from qa4iqi_extraction.constants import MANUFACTURER_FIELD, MANUFACTURER_MODEL_NAME_FIELD, SERIES_DESCRIPTION_FIELD, SERIES_NUMBER_FIELD, SLICE_THICKNESS_FIELD
from harmonization.swin.extract import custom_collate_fn, load_data, CropOnROId
tf.disable_v2_behavior()
import os
import numpy as np
import nibabel as nib
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ToTensord,EnsureTyped
from monai.transforms import Compose, LoadImage, EnsureType
from harmonization.swin.extract import CropOnROId
from monai.data import SmartCacheDataset, DataLoader,ThreadDataLoader
import torch
import tensorflow as tf2
gpus = tf2.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf2.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

centersrois = {'cyst1': [324, 334, 158], 'cyst2' : [189, 278, 185], 'hemangioma': [209, 315, 159], 'metastasis': [111, 271, 140], 'normal1': [161, 287, 149], 'normal2': [154, 229, 169]}

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

def run_inference():

    jsonpath = "/mnt/nas7/data/maria/final_features/expanded_registered_light_dataset_info.json"
    model_dir = '/mnt/nas7/data/maria/final_features/QA4IQI/QA4IQI/'
    model_file = model_dir + 'organs-5c-30fs-acc92-121.meta'

    # Start a new session
    sess = tf.Session()

    # Load the graph
    saver = tf.train.import_meta_graph(model_file)
    saver.restore(sess, tf.train.latest_checkpoint(model_dir))

    # Access the graph
    graph = tf.get_default_graph()
    feature_tensor = graph.get_tensor_by_name('MaxPool3D_1:0')

    #names for input data and dropout:
    x = graph.get_tensor_by_name('x_start:0') 
    keepProb = graph.get_tensor_by_name('keepProb:0')  
    
    
    try:
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

        # Calculer le nombre total de poids
        total_params = sum([sess.run(tf.size(var)) for var in variables])
        print(f"Le nombre total de poids dans le modèle est : {total_params}")
    except:
        print("Error while calculating the number of parameters in the model")
    device_id = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    torch.cuda.set_device(device_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    transforms = Compose([
    LoadImage(ensure_channel_first=True),  # Load image and ensure channel dimension
    EnsureType(),                         # Ensure correct data type
    ])


    datafiles = load_data(jsonpath)
    nifti_dir = "/mnt/nas7/data/reza/registered_dataset_all_doses_pad/"
    datafiles = [f for f in os.listdir(nifti_dir) if f.endswith(".nii.gz")]


    output_dir = "/mnt/nas7/data/maria/final_features/"
    output_file = os.path.join(output_dir, "cnn_features_prueba_full2.csv")

    metadata_csv = "/mnt/nas7/data/maria/final_features/ct-fm/dicom_metadata/dicom_metadata.csv"
    metadata_dict = load_metadata(metadata_csv)

    # Open CSV file once
    with open(output_file, "w", newline="") as csvfile:
        fieldnames = ["FileName", "SeriesNumber", "deepfeatures", "ROI", "SeriesDescription", "ManufacturerModelName", "Manufacturer", "SliceThickness"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for file in tqdm(datafiles, desc="Processing CT scans"):
            file_path = os.path.join(nifti_dir, file)
            image = transforms(file_path)

            file_name = file.replace(".nii.gz", "")
            print(file_name)

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

                #patch = patch.to(device)    # Move tensor to the GPU
                flattened_image = patch.flatten()
                flattened_image = np.array(flattened_image)  # Ensure it's a NumPy array
                flattened_image = flattened_image.reshape(1, -1)  # Reshape to (1, 131072)

                features = sess.run(feature_tensor, feed_dict={x: flattened_image, keepProb: 1.0})

                latentrep = sess.run(tf.reshape(features, [-1])).tolist()

                # Write to CSV
                writer.writerow({
                    "FileName": file,
                    "ROI": roi_label,
                    "deepfeatures": latentrep,
                    "SeriesDescription": series_description,
                    "ManufacturerModelName": manufacturer_model_name,
                    "Manufacturer": manufacturer,
                    "SliceThickness": slicethickness
                })

            #writer.writerow(record)
                 
    print("Done!")


if __name__ == "__main__":
    #test()
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



import os
import numpy as np
import nibabel as nib
from monai.transforms import Compose, LoadImage, EnsureType

# Directorios de entrada y salida
nifti_dir = "/mnt/nas7/data/reza/registered_dataset_all_doses_pad/"
output_dir = "/mnt/nas7/data/maria/final_features/patches/"
os.makedirs(output_dir, exist_ok=True)

# Definir los centros de los ROIs
centersrois = {
    'cyst1': [324, 334, 158], 'cyst2': [189, 278, 185], 'hemangioma': [209, 315, 159],
    'metastasis': [111, 271, 140], 'normal1': [161, 287, 149], 'normal2': [154, 229, 169]
}

# Función para extraer un parche de una imagen
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

    return patch

# Cargar solo las dos primeras imágenes
datafiles = [f for f in os.listdir(nifti_dir) if f.endswith(".nii.gz")][:2]

# Transformaciones para cargar las imágenes
transforms = Compose([
    LoadImage(ensure_channel_first=True),
    EnsureType()
])

# Procesar las imágenes
for file in datafiles:
    file_path = os.path.join(nifti_dir, file)
    image = transforms(file_path)
    file_name = file.replace(".nii.gz", "")
    
    # Extraer y guardar los 6 parches
    for roi_label, center in centersrois.items():
        patch = extract_patch(image, np.array(center))
        patch = patch.squeeze()
        patch_nifti = nib.Nifti1Image(patch.numpy(), affine=np.eye(4))
        print(patch_nifti.shape)
        patch_filename = os.path.join(output_dir, f"{file_name}_{roi_label}.nii.gz")
        nib.save(patch_nifti, patch_filename)
        print(f"Guardado: {patch_filename}")

print("Proceso completado.")

'''