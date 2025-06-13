import SimpleITK as sitk
import os
import csv
import pandas as pd
import re

# Define main directory and subfolders to process
main_folder = "/mnt/nas4/datasets/ToCurate/QA4IQI/FinalDataset-TCIA-MultiCentric/Upl"
subfolders = ["A1", "A2", "B1", "B2", "G1", "G2", "C1", "H2", "D1", "E2", "F1", "E1", "H1"]

# Output directory and CSV file path
output_folder = "/mnt/nas7/data/maria/final_features/ct-fm/dicom_metadata"
csv_filename = os.path.join(output_folder, "dicom_metadata.csv")
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
