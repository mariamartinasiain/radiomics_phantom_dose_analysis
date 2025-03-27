'''
import os
import csv
import json
import glob

# Paths
metadata_csv_path = "/mnt/nas7/data/maria/final_features/ct-fm/dicom_metadata/dicom_metadata.csv"
image_folder = "/mnt/nas7/data/reza/registered_dataset_all_doses_pad"
output_json_path = "/mnt/nas7/data/maria/final_features/dicom_metadata.json"

# Read metadata from CSV
metadata_list = []
with open(metadata_csv_path, mode="r") as file:
    reader = csv.DictReader(file)
    for row in reader:
        metadata_list.append(row)

# Get available image files and map them by SeriesDescription
image_files = glob.glob(os.path.join(image_folder, "*.nii.gz"))
image_map = {}
for img in image_files:
    filename = os.path.basename(img)
    for metadata in metadata_list:
        if metadata["Folder"] in filename:
            image_map[metadata["SeriesDescription"]] = img

# Create JSON structure
json_data = []
for metadata in metadata_list:
    series_description = metadata["SeriesDescription"]
    if series_description in image_map:
        entry = {
            "image": image_map[series_description],
            "info": {
                "SeriesNumber": int(metadata["SeriesNumber"]) if metadata["SeriesNumber"].isdigit() else metadata["SeriesNumber"],
                "SeriesDescription": metadata["SeriesDescription"],
                "Manufacturer": metadata["Manufacturer"],
                "ManufacturerModelName": metadata["ManufacturerModelName"],
                "SliceThickness": float(metadata["SliceThickness"]) if metadata["SliceThickness"].replace('.', '', 1).isdigit() else None,
                "StudyDescription": metadata["StudyDescription"],
                "StudyID": metadata["StudyID"]
            },
            "roi_label": None  # You can update this if needed
        }
        json_data.append(entry)

# Save to JSON
with open(output_json_path, "w") as json_file:
    json.dump(json_data, json_file, indent=4)

print(f"JSON file saved at {output_json_path}")
'''


import json

# File paths
file_paths = [
    "/mnt/nas7/data/maria/final_features/expanded_registered_light_dataset_info.json",
    #"/mnt/nas7/data/maria/final_features/dicom_metadata.json"
]

# Doses to check
doses = ["1mGy", "3mGy", "6mGy", "10mGy", "14mGy"]

for file_path in file_paths:
    with open(file_path, "r") as f:
        data = json.load(f)

        # Count total dictionaries
        print(f"{file_path}: {len(data)} dictionaries")

        # Count unique SeriesNumber
        unique_series_numbers = set(entry["info"]["SeriesNumber"] for entry in data)
        print(f"Unique SeriesNumber: {len(unique_series_numbers)}")

        # Count occurrences of each dose
        for dose in doses:
            count_dose = sum(1 for entry in data if dose in entry["info"]["SeriesDescription"])
            print(f"Occurrences of '{dose}' in SeriesDescription: {count_dose / 6}")

        print()  # Empty line for better readability


import csv

# Path to your CSV file
file_path = '/home/reza/radiomics_phantom/final_features_doses/features_pyradiomics.csv'
#file_path = '/home/reza/radiomics_phantom/final_features_doses/features_swin.csv'

# Open the CSV file
with open(file_path, "r") as f:
    reader = csv.DictReader(f)  # Use DictReader to read rows as dictionaries
    
    # Extract the SeriesNumber from each row and add it to a set for uniqueness
    unique_series_numbers = set(entry["SeriesDescription"] for entry in reader)

# Print the count of unique SeriesNumber
print(f"Unique SeriesNumber: {len(unique_series_numbers)}")
