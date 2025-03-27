import os
import logging
import pandas as pd
import radiomics
from collections import OrderedDict
import json
import nibabel as nib

from qa4iqi_extraction.constants import (
    DIAGNOSTICS_FEATURES_PREFIX,
    PARAMETER_FILE_NAME,
)

logger = logging.getLogger()

def load_metadata(json_filename):
    """Loads the metadata JSON and returns a list of metadata."""
    with open(json_filename, 'r') as f:
        metadata = json.load(f)  # Parse the JSON file
    return metadata


def extract_features(image_path, rois_paths, metadata_json):
    params_file_path = PARAMETER_FILE_NAME

    radiomics.logger.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())
    radiomics.setVerbosity(
        getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper())
    )

    extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(params_file_path)

    file_name = os.path.basename(image_path).replace(".nii.gz", "")
    print(file_name)

    # Load metadata from JSON file
    metadata_list = load_metadata(metadata_json)

    all_features = []

    for roi_path in rois_paths:

        # Extract the ROI name using
        roi_filename = os.path.basename(roi_path)
        roi_name = roi_filename.split('-')[-1].replace('.nii.gz', '')

        # Find the metadata entry that matches the image and roi_name
        metadata_entry = None
        for item in metadata_list:
            image_filename = os.path.basename(item["image"]).replace(".nii.gz", "")  # Extract filename without extension
            if file_name == image_filename and item["roi_label"] == roi_name:
                metadata_entry = item
                break

        print('Processing file:', image_filename)
        print('Processing ROI:', roi_name)

        if metadata_entry is None:
            print(f"Warning: No matching metadata for ROI '{roi_name}' in image '{file_name}'")
            continue  # Skip this ROI if no matching metadata is found

        # Extract metadata from the matching entry
        metadata_info = metadata_entry["info"]

        # Extract the features
        roi_features = extractor.execute(image_path, roi_path, label=1)

        # Filter out diagnostics features which are not wanted
        roi_features = OrderedDict(
            {k: v.item() for k, v in roi_features.items() if not k.startswith(DIAGNOSTICS_FEATURES_PREFIX)}
        )


        # Add metadata to the features
        roi_features["FileName"] = file_name
        roi_features["ROI"] = roi_name
        roi_features["SeriesNumber"] = metadata_info.get("SeriesNumber", "Unknown")
        roi_features["SeriesDescription"] = metadata_info.get("SeriesDescription", "Unknown")
        roi_features["ManufacturerModelName"] = metadata_info.get("ManufacturerModelName", "Unknown")
        roi_features["Manufacturer"] = metadata_info.get("Manufacturer", "Unknown")
        roi_features["SliceThickness"] = metadata_info.get("SliceThickness", "Unknown")
        roi_features["StudyDescription"] = metadata_info.get("StudyDescription", "Unknown")
        roi_features["StudyID"] = metadata_info.get("StudyID", "Unknown")

        all_features.append(roi_features)

    # Convert to DataFrame
    features_df = pd.DataFrame.from_records(all_features)

    return features_df

