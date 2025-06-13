import os
import logging
import json

from glob import glob
from pydicom import dcmread
from tqdm import tqdm

from radiomics.constants import (
    FIELD_NAME_IMAGE,
    FIELD_NAME_SEG,
    MODALITY_CT,
    MODALITY_SEG,
)

logger = logging.getLogger()


def identify_images_rois(folder):
    logger.info("Identifying Image & ROI pairs...")

    # Check if map is already available
    #map_file_path = f"{folder}/fstudies_map.json"
    map_file_path = '/mnt/nas7/data/maria/final_features/pyradiomics_extraction/fstudies_map.json'


    if os.path.exists(map_file_path):
        logger.info("Existing mapping found!")
        with open(map_file_path, "r") as map_file:
            study_folders_map = json.load(map_file)
            return study_folders_map

    series_folders = [root for root, _, _ in os.walk(folder)]
    
    study_folders_map = {}

    # Read one file from each folder to identify Image -> ROI pairs
    for series_folder in tqdm(series_folders):
        dicom_files = glob(f"{series_folder}/*")
        dicom_files = [f for f in dicom_files if os.path.isfile(f)]
        #print("folder", folder)
        #print("series_folder", series_folder)
        #print("dicom_files", dicom_files)
        if len(dicom_files) == 0:
            continue
        
        first_dicom_file = dicom_files[0]
        ds = dcmread(first_dicom_file, defer_size="1 KB", stop_before_pixels=True)

        #print("ds", ds)
        # Check if it's the image or the ROIs
        study_uid = ds.StudyInstanceUID
        modality = ds.Modality
        print("modality", modality)
        logger.debug(f"Modality: {modality}")

        if study_uid not in study_folders_map:
            study_folders_map[study_uid] = {}

        if modality == MODALITY_CT:
            study_folders_map[study_uid][FIELD_NAME_IMAGE] = os.path.dirname(
                first_dicom_file
            )
        elif modality == MODALITY_SEG:
            study_folders_map[study_uid][FIELD_NAME_SEG] = first_dicom_file
        else:
            raise ValueError(f"Modality {modality} is not supported")

    with open(map_file_path, "w") as map_file:
        json.dump(study_folders_map, map_file)

    logger.debug(f"dicom_folders_map structure: {study_folders_map}")


    return study_folders_map
