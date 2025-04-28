import SimpleITK as sitk
import os
import csv

# Define main directory and subfolders to process
main_folder = "/mnt/nas4/datasets/ToCurate/QA4IQI/FinalDataset-TCIA-MultiCentric/Upl"
subfolders = ["A1", "A2", "B1", "B2", "G1", "G2", "C1", "H2", "D1", "E2", "F1", "E1", "H1"]

# Output directory and CSV file path
output_folder = "/mnt/nas7/data/maria/final_features/ct-fm/dicom_metadata"
csv_filename = os.path.join(output_folder, "dicom_metadata.csv")

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

# Define metadata keys to extract
metadata_keys = {
    "SeriesDescription": "0008|103e",
    "SeriesNumber": "0020|0011", 
    "ManufacturerModelName": "0008|1090",
    "Manufacturer": "0008|0070",
    "StudyDescription": "0008|1030",
    "StudyID": "0020|0010",
    "SliceThickness": "0018|0050",
}

# Open CSV file for writing
with open(csv_filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    
    # Write the header row
    writer.writerow(["Folder"] + list(metadata_keys.keys()))

    # Loop through each subfolder
    for subfolder in subfolders:
        subfolder_path = os.path.join(main_folder, subfolder)
        
        if not os.path.isdir(subfolder_path):
            print(f"Skipping {subfolder} (not a directory)")
            continue
        
        print(f"\nProcessing folder: {subfolder}")

        # Loop through all subfolders inside the main subfolder
        for subfile in sorted(os.listdir(subfolder_path)):
            subfile_path = os.path.join(subfolder_path, subfile)
            
            if not os.path.isdir(subfile_path):
                continue  # Skip if not a folder
            
            print(f"\nProcessing folder: {subfile}")

            # Reinitialize the reader for each iteration
            reader = sitk.ImageSeriesReader()

            # Get the list of DICOM series in the subfolder
            series_ids = reader.GetGDCMSeriesIDs(subfile_path)

            if not series_ids:
                print(f"No DICOM series found in {subfile}.")
                continue

            # Process only the first series found in the folder
            series_file_names = reader.GetGDCMSeriesFileNames(subfile_path, series_ids[0])
            reader.SetFileNames(series_file_names)

            # Ensure metadata is extracted
            reader.MetaDataDictionaryArrayUpdateOn()
            reader.LoadPrivateTagsOn()

            # Read the DICOM series
            image = reader.Execute()

            # Extract metadata from the first slice
            metadata_values = {}
            try:
                for key_name, dicom_key in metadata_keys.items():
                    if reader.HasMetaDataKey(0, dicom_key):
                        metadata_values[key_name] = reader.GetMetaData(0, dicom_key)
                    else:
                        metadata_values[key_name] = "N/A"  # If metadata is missing
            except RuntimeError:
                print(f"  - Error reading metadata in {subfile}.")
                continue

            # Write data to CSV
            writer.writerow([subfile] + [metadata_values[key] for key in metadata_keys])

print(f"\nMetadata extraction complete. Data saved in: {csv_filename}")

'''

import pandas as pd
import re


# File paths
file_path = "/mnt/nas7/data/maria/final_features/ct-fm/dicom_metadata/dicom_metadata.csv"
swinunetr_path = "/mnt/nas7/data/maria/final_features/features_pyradiomics_full.csv"

# Load CSV files
metadata = pd.read_csv(file_path)
swinunetr_data = pd.read_csv(swinunetr_path)

# Extract SeriesNumber columns
metadata_series = set(metadata["SeriesNumber"].dropna().astype(int))


def extract_number(tensor_str):
    match = re.search(r'\d+', str(tensor_str))
    return int(match.group()) if match else None

def extract_mg_value(series_description):
    """Extracts dose (mGy value) from the SeriesDescription column if followed by 'mGy'."""
    match = re.search(r'(\d+)mGy', str(series_description))
    return int(match.group(1)) if match else None

# Convert SeriesNumber to integer (if not already)
swinunetr_data["SeriesNumber"] = swinunetr_data["SeriesNumber"].dropna().astype(int)

# Extract numerical doses from SeriesDescription
swinunetr_data["Dose"] = swinunetr_data["SeriesDescription"].apply(extract_mg_value)

# Define dose values to analyze
dose_values_to_analyze = [3, 14]

# Count occurrences of #1 to #10 in SeriesDescription for the selected doses
for dose_value in dose_values_to_analyze:
    print(f"\nCounts of #1 to #10 in SeriesDescription for Dose = {dose_value} mGy:")
    
    filtered_data = swinunetr_data.loc[swinunetr_data["Dose"] == dose_value, "SeriesDescription"]
    
    count_dict = {f"#{i}": 0 for i in range(1, 11)}
    
    for desc in filtered_data.dropna():
        match = re.search(r"#(\d+)", desc)
        if match:
            num = f"#{match.group(1)}"
            if num in count_dict:
                count_dict[num] += 1
    
    for key, value in count_dict.items():
        print(f"{key}: {value}")

# Check for duplicate SeriesDescription values
duplicate_series = swinunetr_data["SeriesDescription"].value_counts()
duplicates = duplicate_series[duplicate_series < 6]

if not duplicates.empty:
    print("\nRepeated SeriesDescription values:")
    print(duplicates)
    print(len(duplicates))
else:
    print("\nNo repeated SeriesDescription values found.")

# Filter data for Dose = 3 mGy
dose_3_data = swinunetr_data[swinunetr_data["Dose"] == 3]

# Exclude rows where SeriesDescription contains 'DL'
dose_3_data = dose_3_data[~dose_3_data["SeriesDescription"].str.contains("DL", na=False)]

# ROI types to count
roi_types = ['normal1', 'normal2', 'cyst1', 'cyst2', 'hemangioma', 'metastasis']

# Define dose values to analyze
dose_values_to_analyze = [3, 14]

# Count occurrences of #1 to #10 in SeriesDescription for the selected doses
for dose_value in dose_values_to_analyze:
    print(f"\nCounts of #1 to #10 in SeriesDescription for Dose = {dose_value} mGy:")

    # Filter data for the selected dose and exclude rows where SeriesDescription contains 'DL'
    filtered_data = swinunetr_data.loc[(swinunetr_data["Dose"] == dose_value) &
                                       (~swinunetr_data["SeriesDescription"].str.contains("DL", na=False)), "SeriesDescription"]
    
    # Initialize count dictionary for #1 to #10
    count_dict = {f"#{i}": 0 for i in range(1, 11)}
    
    # Count occurrences of #1 to #10
    for desc in filtered_data.dropna():
        match = re.search(r"#(\d+)", desc)
        if match:
            num = f"#{match.group(1)}"
            if num in count_dict:
                count_dict[num] += 1
    
    # Print the counts for each #1 to #10
    for key, value in count_dict.items():
        print(f"{key}: {value}")


swinunetr_data["SeriesNumber"] = swinunetr_data["SeriesNumber"].dropna().astype(int)
swinunetr_series = set(swinunetr_data["SeriesNumber"].dropna())

missing_in_swinunetr = metadata_series - swinunetr_series

print("SeriesNumbers missing in SwinUNETR features file:", sorted(missing_in_swinunetr))
print(len(missing_in_swinunetr))



import pandas as pd
import os
import glob

# Directorio donde están los archivos
path = "/mnt/nas7/data/maria/final_features/final_features_complete"

# Buscar archivos que terminan en '6rois.csv'
csv_files = glob.glob(os.path.join(path, "*6rois.csv"))

for file in csv_files:
    df = pd.read_csv(file)

    # Reemplazar valores en la columna 'ROI'
    df["ROI"] = df["ROI"].replace({"normal1": "normal", "normal2": "normal",
                                   "cyst1": "cyst", "cyst2": "cyst"})

    # Crear nuevo nombre de archivo
    new_file = file.replace("6rois.csv", "4rois.csv")

    # Guardar el nuevo archivo
    df.to_csv(new_file, index=False)

    print(f"Archivo guardado: {new_file}")
'''