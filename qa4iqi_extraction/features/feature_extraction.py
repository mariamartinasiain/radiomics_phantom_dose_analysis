import logging
import tempfile
import pandas as pd

from tqdm import tqdm
from qa4iqi_extraction.constants import (
    FIELD_NAME_IMAGE,
    FIELD_NAME_SEG,
    SERIES_DESCRIPTION_FIELD,
    SERIES_NUMBER_FIELD,
    STUDY_UID_FIELD,
    MANUFACTURER_FIELD,
    MANUFACTURER_MODEL_NAME_FIELD,
    SLICE_THICKNESS_FIELD,
    SLICE_SPACING_FIELD,
)
from qa4iqi_extraction.features.extract_features import extract_features

from qa4iqi_extraction.utils.nifti2 import convert_to_nifti

logging.basicConfig(filename="std.log", 
					format='%(asctime)s %(message)s', 
					filemode='w') 


logger = logging.getLogger()


def run_feature_extraction(dicom_folders_map):
    logger.info("Running feature extraction...")

    i = 0
    all_features_df = []
    for study_uid, dicom_image_mask in tqdm(
        dicom_folders_map.items(), desc="Processing all DICOM studies"
    ):
        logger.debug(f"dicom_image_mask structure: {dicom_image_mask}")
        with tempfile.TemporaryDirectory(prefix=study_uid) as tmp_dir:
            dirr = "./"
            try : 
                nifti_image_path, nifti_roi_paths, dicom_info = convert_to_nifti(
                    dicom_image_mask, dirr
                )
            except Exception as e:
                logger.error(f"Error converting study {study_uid}: {e} ... From file {dicom_image_mask} where image is {dicom_image_mask[FIELD_NAME_IMAGE]} and seg is {dicom_image_mask[FIELD_NAME_SEG]}. Skipping...")
                print(f"Error converting study {study_uid}: {e} ... From file {dicom_image_mask}. Skipping...")
                continue
            
            logger.debug(f"Done converting study {study_uid}")
            
            print("finished converting study from file", dicom_image_mask)

            features_df = extract_features(
                nifti_image_path, nifti_roi_paths, dicom_info[SERIES_DESCRIPTION_FIELD]
            )

            features_df.insert(
                0, SERIES_DESCRIPTION_FIELD, dicom_info[SERIES_DESCRIPTION_FIELD]
            )
            features_df.insert(0, SERIES_NUMBER_FIELD, dicom_info[SERIES_NUMBER_FIELD])
            features_df.insert(0, STUDY_UID_FIELD, study_uid)
            features_df.insert(0,MANUFACTURER_FIELD, dicom_info[MANUFACTURER_FIELD])
            features_df.insert(0,MANUFACTURER_MODEL_NAME_FIELD, dicom_info[MANUFACTURER_MODEL_NAME_FIELD])
            features_df.insert(0,SLICE_THICKNESS_FIELD, dicom_info[SLICE_THICKNESS_FIELD])
            features_df.insert(0,SLICE_SPACING_FIELD, dicom_info[SLICE_SPACING_FIELD])

            all_features_df.append(features_df)

        i += 1

    # Concatenate all dataframes
    if not all_features_df:
        return pd.DataFrame()
    concatenated_features_df = pd.concat(all_features_df, ignore_index=True)

    # Sort by series number
    concatenated_features_df = concatenated_features_df.sort_values(SERIES_NUMBER_FIELD)

    return concatenated_features_df
