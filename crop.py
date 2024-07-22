import pydicom
import pydicom_seg
import numpy as np
import SimpleITK as sitk
import os

def pad_segmentation(seg, target_shape, start_index, end_index):
    padded_seg = np.zeros(target_shape, dtype=seg.dtype)
    padded_seg[start_index:end_index, :, :] = seg
    return padded_seg

def find_closest_index(array, value):
    if not array:
        raise ValueError("Input array is empty")
    return np.argmin(np.abs(np.array(array) - value))

def crop_volume(mask_file, output_path, crop_coords, reference_dicom_folder):
    print(f"Mask file: {mask_file}")
    print(f"Reference DICOM folder: {reference_dicom_folder}")
    
    # Read the DICOM segmentation file
    dicom_seg = pydicom.dcmread(mask_file)
    reader = pydicom_seg.SegmentReader()
    result = reader.read(dicom_seg)

    # Read reference DICOM files
    dicom_files = [os.path.join(reference_dicom_folder, f) for f in os.listdir(reference_dicom_folder) if f.isdigit()]
    dicom_files.sort()
    print(f"Number of potential DICOM files found: {len(dicom_files)}")

    dicom_datasets = []
    all_instance_z_locations = []
    for f in dicom_files:
        try:
            ds = pydicom.dcmread(f)
            dicom_datasets.append(ds)
            if hasattr(ds, 'ImagePositionPatient'):
                all_instance_z_locations.append(float(ds.ImagePositionPatient[-1]))
            else:
                print(f"Warning: ImagePositionPatient not found in file {f}")
        except Exception as e:
            print(f"Error reading file {f}: {e}")

    print(f"Number of valid DICOM datasets: {len(dicom_datasets)}")
    print(f"Number of Z locations found: {len(all_instance_z_locations)}")

    if not all_instance_z_locations:
        raise ValueError("No valid Z locations found in DICOM files")

    all_referenced_z_locations = [
        float(f.PlanePositionSequence[0].ImagePositionPatient[-1])
        for f in dicom_seg.PerFrameFunctionalGroupsSequence
    ]
    all_referenced_z_locations = np.unique(all_referenced_z_locations)

    min_referenced_z_location = min(all_referenced_z_locations)

    # Find the closest index instead of exact match
    starting_index_global = find_closest_index(all_instance_z_locations, min_referenced_z_location)
    ending_index_global = starting_index_global + len(all_referenced_z_locations)

    print(f"Starting index: {starting_index_global}, Ending index: {ending_index_global}")

    # Process all segments into a single mask
    full_mask = np.zeros((len(dicom_datasets), 512, 512), dtype=np.uint8)
    
    for segment_number in result.available_segments:
        segmentation_image_data = result.segment_data(segment_number)

        # Change axes to match DICOM
        seg = np.swapaxes(segmentation_image_data, 0, -1)

        # Pad segmentation to match DICOM dimensions
        padded_seg = pad_segmentation(
            seg, full_mask.shape, starting_index_global, ending_index_global
        )

        # Add this segment to the full mask
        full_mask = np.logical_or(full_mask, padded_seg).astype(np.uint8)

    # Convert to SimpleITK image for cropping
    sitk_image = sitk.GetImageFromArray(full_mask)

    # Save the full mask as NIFTI
    full_mask_path = os.path.splitext(output_path)[0] + "_full.nii.gz"
    sitk.WriteImage(sitk_image, full_mask_path)
    print(f"Full mask saved as {full_mask_path}")

    # Adjust crop coordinates for (z, y, x) order
    crop_start = [crop_coords[0], crop_coords[2], crop_coords[4]]
    crop_size = [crop_coords[1] - crop_coords[0],
                 crop_coords[3] - crop_coords[2],
                 crop_coords[5] - crop_coords[4]]

    # Ensure crop is within image bounds
    image_size = sitk_image.GetSize()
    crop_start = [max(0, min(s, image_size[i] - 1)) for i, s in enumerate(crop_start)]
    crop_size = [min(s, image_size[i] - crop_start[i]) for i, s in enumerate(crop_size)]

    print(f"Image size: {image_size}")
    print(f"Crop start: {crop_start}")
    print(f"Crop size: {crop_size}")

    cropped_image = sitk.Crop(sitk_image, crop_start, crop_size)

    # Save the cropped mask as NIFTI
    cropped_mask_path = os.path.splitext(output_path)[0] + "_cropped.nii.gz"
    sitk.WriteImage(cropped_image, cropped_mask_path)
    print(f"Cropped mask saved as {cropped_mask_path}")

    print(f"Cropped image size: {cropped_image.GetSize()}")

def main():
    base_path = "/mnt/nas4/datasets/ToCurate/QA4IQI/FinalDataset-TCIA-MultiCentric/Upl/A1"
    reference_volume = "A1_174008_691000_SOMATOM_Definition_Edge_ID23_Harmonized_10mGy_IR_NrFiles_343"
    mask_file = os.path.join(base_path, reference_volume, "mask", f"{reference_volume}.dcm")
    reference_dicom_folder = os.path.join(base_path, reference_volume)

    # Crop coordinates [z_start, z_end, y_start, y_end, x_start, x_end]
    crop_coords = [13, 323, 120, 395, 64, 445]

    # Output path for the cropped mask
    output_path = os.path.join(base_path, f"{reference_volume}_cropped_mask.dcm")

    # Crop the mask
    crop_volume(mask_file, output_path, crop_coords, reference_dicom_folder)

if __name__ == "__main__":
    main()