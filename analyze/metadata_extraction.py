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
file_path = "/mnt/nas7/data/maria/final_features/ct-fm/dicom_metadata/dicom_metadata2.csv"
cnn_path = "/mnt/nas7/data/maria/final_features/features_cnn_full.csv"
swinunetr_path = "/mnt/nas7/data/maria/final_features/features_swinunetr_full.csv"

# Load CSV files
metadata = pd.read_csv(file_path)
cnn_data = pd.read_csv(cnn_path)
swinunetr_data = pd.read_csv(swinunetr_path)

# Extract SeriesNumber columns
metadata_series = set(metadata["SeriesNumber"].dropna().astype(int))

def extract_number(tensor_str):
    match = re.search(r'\d+', str(tensor_str))
    return int(match.group()) if match else None

cnn_data["SeriesNumber"] = cnn_data["SeriesNumber"].dropna().apply(extract_number)
swinunetr_data["SeriesNumber"] = swinunetr_data["SeriesNumber"].dropna().apply(extract_number)

cnn_series = set(cnn_data["SeriesNumber"].dropna())
swinunetr_series = set(swinunetr_data["SeriesNumber"].dropna())

# Find missing series numbers
missing_in_cnn = metadata_series - cnn_series
missing_in_swinunetr = metadata_series - swinunetr_series

# Count occurrences of each SeriesNumber
cnn_counts = cnn_data["SeriesNumber"].value_counts()
swinunetr_counts = swinunetr_data["SeriesNumber"].value_counts()

# Print missing series numbers
print("SeriesNumbers missing in CNN features file:", sorted(missing_in_cnn))
print(len(missing_in_cnn))
print("SeriesNumbers missing in SwinUNETR features file:", sorted(missing_in_swinunetr))
print(len(missing_in_swinunetr))

# Find SeriesNumbers with counts < 6
cnn_less_than_6 = cnn_counts[cnn_counts > 6]
swinunetr_less_than_6 = swinunetr_counts[swinunetr_counts > 6]

# Print the Series numbers with counts < 6
print("\nCNN Series with less than 6 counts:")
for series_number in sorted(cnn_less_than_6.index):
    print(f"SeriesNumber {series_number} has {cnn_less_than_6[series_number]} counts")

print("\nSwinUNETR Series with less than 6 counts:")
for series_number in sorted(swinunetr_less_than_6.index):
    print(f"SeriesNumber {series_number} has {swinunetr_less_than_6[series_number]} counts")

# Calculate missing counts by subtracting less than 6 counts from 6
cnn_missing_count = cnn_counts[cnn_counts < 6].apply(lambda x: 6 - x).sum()
swinunetr_missing_count = swinunetr_counts[swinunetr_counts < 6].apply(lambda x: 6 - x).sum()

# Print total missing counts
print("\nTotal missing counts in CNN features file:", cnn_missing_count)
print("Total missing counts in SwinUNETR features file:", swinunetr_missing_count)
'''


