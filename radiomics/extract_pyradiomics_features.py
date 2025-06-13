import os
import logging
import pandas as pd
import radiomics
from collections import OrderedDict
import json
import nibabel as nib
from glob import glob

from radiomics.constants import (
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
        roi_name = roi_filename.split('_')[-1].replace('.nii.gz', '')

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


def extract_features_10regions(image_path, masks_folder, metadata_json):
    params_file_path = PARAMETER_FILE_NAME

    radiomics.logger.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())
    radiomics.setVerbosity(
        getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper())
    )

    extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(params_file_path)

    file_name = os.path.basename(image_path).replace(".nii.gz", "")
    print("Processing image:", file_name)

    # Load metadata
    metadata_list = load_metadata(metadata_json)
    all_features = []

    # Load masks
    mask_paths = sorted(glob(os.path.join(masks_folder, "*.nii.gz")))
    for mask_path in mask_paths:
        mask_filename = os.path.basename(mask_path).replace('.nii.gz', '')  # e.g., cyst1_01

        # Parse ROI name and image ID (e.g., cyst1_01 → ROI = cyst1, id = 01)
        if '_' not in mask_filename:
            print(f"Skipping invalid mask filename format: {mask_filename}")
            continue
        roi_name, id = mask_filename.rsplit('_', 1)

        # Match metadata
        metadata_entry = None
        for item in metadata_list:
            image_filename = os.path.basename(item["image"]).replace(".nii.gz", "")
            if file_name == image_filename and item["roi_label"] == roi_name:
                metadata_entry = item
                break

        print("Processing ROI:", roi_name, "| Patch number:", id)

        if metadata_entry is None:
            print(f"Warning: No matching metadata for ROI '{roi_name}' in image '{file_name}'")
            continue

        metadata_info = metadata_entry["info"]

        # Extract features
        roi_features = extractor.execute(image_path, mask_path, label=1)

        # Filter out diagnostic features
        roi_features = OrderedDict(
            {k: v.item() if hasattr(v, 'item') else v for k, v in roi_features.items() if not k.startswith(DIAGNOSTICS_FEATURES_PREFIX)}
        )

        # Add metadata
        roi_features["FileName"] = file_name
        roi_features["ROI"] = roi_name
        roi_features["PatchNumber"] = id
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


def extract_features_org(image_path, mask_paths, organ_names):
    params_file_path = PARAMETER_FILE_NAME

    radiomics.logger.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())
    radiomics.setVerbosity(
        getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper())
    )

    extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(params_file_path)

    file_name = os.path.basename(image_path).replace(".nii.gz", "")
    print("Processing image:", file_name)

    all_features = []

    # Check mask_paths and organ_names are same length
    if len(mask_paths) != len(organ_names):
        raise ValueError("mask_paths and organ_names must have the same length")

    for mask_path, organ in zip(mask_paths, organ_names):
        print('Organ:', organ)

        try:
            roi_features = extractor.execute(image_path, mask_path, label=1)
        except Exception as e:
            print(f"Error extracting features for organ {organ}: {e}")
            continue

        # Filter out diagnostic features
        roi_features = OrderedDict(
            {k: v.item() if hasattr(v, 'item') else v for k, v in roi_features.items() if not k.startswith(DIAGNOSTICS_FEATURES_PREFIX)}
        )

        # Add metadata columns
        roi_features["FileName"] = file_name
        roi_features["ROI"] = organ

        all_features.append(roi_features)

    features_df = pd.DataFrame.from_records(all_features)
    return features_df

