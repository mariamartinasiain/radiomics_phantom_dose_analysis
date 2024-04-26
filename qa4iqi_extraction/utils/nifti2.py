from glob import glob
import json
import os
import numpy as np
import nibabel as nib
import dcmstack
import pydicom
import pydicom_seg


from qa4iqi_extraction.constants import (
    FIELD_NAME_IMAGE,
    FIELD_NAME_SEG,
    FOLDER_NAME_IMAGE,
    FOLDER_NAME_ROIS,
    SERIES_DESCRIPTION_FIELD,
    SERIES_NUMBER_FIELD,
    MANUFACTURER_FIELD,
    MANUFACTURER_MODEL_NAME_FIELD,
    SLICE_THICKNESS_FIELD,
    SLICE_SPACING_FIELD,
)


def convert_to_nifti(dicom_image_mask, nifti_dir):
    print("Convert to nifti DEUX")
    output_folder_image = os.path.join(nifti_dir, FOLDER_NAME_IMAGE)
    output_folder_rois = os.path.join(nifti_dir, FOLDER_NAME_ROIS)
    output_file_seg_prefix = "segmentation"
    output_file_seg_suffix = ".nii.gz"

    os.makedirs(output_folder_image, exist_ok=True)
    print("output_folder_image", output_folder_image)
    os.makedirs(output_folder_rois, exist_ok=True)
    print("output_folder_rois", output_folder_rois)
    print("output folder image", output_folder_image)

    dicom_image_folder = dicom_image_mask[FIELD_NAME_IMAGE]
    dicom_seg_file = dicom_image_mask[FIELD_NAME_SEG]

    # Read DICOM image & convert to NIfTI
    dicom_files = glob(f"{dicom_image_folder}/*",include_hidden=True)
    dicom_files = [f for f in dicom_files if (not os.path.isdir(f))]
    dicom_datasets = [pydicom.dcmread(f,force=True) for f in dicom_files]
    dicom_datasets = sorted(dicom_datasets, key=lambda ds: -ds.InstanceNumber)

    # Store useful DICOM metadata
    dicom_info = {}
    dicom_info[SERIES_NUMBER_FIELD] = None
    dicom_info[SERIES_DESCRIPTION_FIELD] = None
    dicom_info[MANUFACTURER_FIELD] = None
    dicom_info[MANUFACTURER_MODEL_NAME_FIELD] = None
    dicom_info[SLICE_THICKNESS_FIELD] = None
    dicom_info[SLICE_SPACING_FIELD] = None
    
    try:
        dicom_info[SERIES_NUMBER_FIELD] = dicom_datasets[0].SeriesNumber
    except Exception as e:
        print(f"Error reading DICOM metadata: {e}")
    try:
        dicom_info[SERIES_DESCRIPTION_FIELD] = dicom_datasets[0].SeriesDescription
    except Exception as e:
        print(f"Error reading DICOM metadata: {e}")
    try:
        dicom_info[MANUFACTURER_FIELD] = dicom_datasets[0].Manufacturer
    except Exception as e:
        print(f"Error reading DICOM metadata: {e}")
    try:
        dicom_info[MANUFACTURER_MODEL_NAME_FIELD] = dicom_datasets[0].ManufacturerModelName
    except Exception as e:
        print(f"Error reading DICOM metadata: {e}")
    try:
        dicom_info[SLICE_THICKNESS_FIELD] = dicom_datasets[0].SliceThickness
    except Exception as e:
        print(f"Error reading DICOM metadata: {e}")
    try:
        dicom_info[SLICE_SPACING_FIELD] = dicom_datasets[0].SpacingBetweenSlices
    except Exception as e:
        print(f"Error reading DICOM metadata: {e}")

    stack = dcmstack.DicomStack()
    for ds in dicom_datasets:
        stack.add_dcm(ds)
    nii = stack.to_nifti()
    unique_id = os.path.basename(dicom_image_folder)
    nifti_image_path = f"{output_folder_image}/image_{unique_id}.nii.gz"
    print("nifti_image_path", nifti_image_path)
    nii.to_filename(nifti_image_path, dtype=np.uint16)

    # Read DICOM SEG & convert to NIfTI
    dicom_seg = pydicom.dcmread(dicom_seg_file)
    reader = pydicom_seg.SegmentReader()
    result = reader.read(dicom_seg)

    # Get index to label mapping
    segment_labels = [s.SegmentLabel for s in dicom_seg.SegmentSequence]

    # Find smallest patient Z position to define starting index
    all_instance_z_locations = [
        float(ds.ImagePositionPatient[-1]) for ds in dicom_datasets
    ]

    all_referenced_z_locations = [
        float(f.PlanePositionSequence[0].ImagePositionPatient[-1])
        for f in dicom_seg.PerFrameFunctionalGroupsSequence
    ]
    all_referenced_z_locations = np.unique(all_referenced_z_locations)

    min_referenced_z_location = min(all_referenced_z_locations)

    starting_index_global = all_instance_z_locations.index(min_referenced_z_location)
    ending_index_global = starting_index_global + len(all_referenced_z_locations)

    # Write out each ROI to a separate file (simpler for pyradiomics extraction)
    nifti_roi_paths = []
    for segment_number in result.available_segments:
        segmentation_image_data = result.segment_data(segment_number)

        # change axes to match dicom
        seg = np.fliplr(np.swapaxes(segmentation_image_data, 0, -1))

        # pad segmentation to match dicom dimensions
        padded_seg = pad_segmentation(
            seg, stack.shape, starting_index_global, ending_index_global
        )

        padded_seg_image = nib.nifti1.Nifti1Image(padded_seg, nii.affine, nii.header)

        nifti_roi_path = f"{output_folder_rois}/{output_file_seg_prefix}_{unique_id}-{segment_number}-{segment_labels[segment_number - 1]}{output_file_seg_suffix}"
        nifti_roi_paths.append(nifti_roi_path)

        padded_seg_image.to_filename(nifti_roi_path, dtype=np.uint8)


    # JSON update
    json_filename = os.path.join(nifti_dir, "dataset_info2.json")
    if os.path.exists(json_filename):
        with open(json_filename, 'r') as json_file:
            json_data = json.load(json_file)
    else:
        json_data = []

    """ new_data_entry = {
        "image": nifti_image_path,
        "rois": [{"path": roi_path, "roi": roi_path.split('-')[-1].replace(output_file_seg_suffix, '')} for roi_path in nifti_roi_paths],
        "info": dicom_info  # Assuming dicom_info contains the additional DICOM metadata including UID
    } """
    for roi_path in nifti_roi_paths:
        new_data_entry = {
            "image": nifti_image_path,
            "roi": roi_path,
            "roi_label": roi_path.split('-')[-1].replace(output_file_seg_suffix, ''),
            "info": dicom_info  # Assuming dicom_info contains the additional DICOM metadata including UID
        }
        json_data.append(new_data_entry)

    with open(json_filename, 'w') as json_file:
        json.dump(json_data, json_file)

    return nifti_image_path, nifti_roi_paths, dicom_info


def pad_segmentation(segmentation, reference_image_shape, starting_index, ending_index):
    padded_seg = np.zeros(reference_image_shape, dtype=np.uint8)

    padded_seg[:, :, starting_index:ending_index] = segmentation

    return padded_seg
