import pydicom
import pydicom_seg
import numpy as np
import SimpleITK as sitk
import os

def pad_segmentation(seg, target_shape, start_index, end_index):
    padded_seg = np.zeros(target_shape, dtype=seg.dtype)
    padded_seg[:, :, start_index:end_index] = seg
    return padded_seg

def find_closest_index(array, value):
    if not array:
        raise ValueError("Input array is empty")
    return np.argmin(np.abs(np.array(array) - value))

def crop_volume(mask_file, output_path, crop_coords, reference_dicom_folder):
    # Read the DICOM segmentation file
    dicom_seg = pydicom.dcmread(mask_file)
    reader = pydicom_seg.SegmentReader()
    result = reader.read(dicom_seg)

    # Read reference DICOM files
    dicom_files = sorted([os.path.join(reference_dicom_folder, f) for f in os.listdir(reference_dicom_folder) if f.isdigit()])
    
    # Read the original volume
    original_volume = sitk.ReadImage(dicom_files)
    
    # Save the original volume as NIFTI
    volume_path = os.path.splitext(output_path)[0] + "_volume.nii.gz"
    sitk.WriteImage(original_volume, volume_path)
    print(f"Original volume saved as {volume_path}")

    # Process the segmentation mask
    full_mask = np.zeros(original_volume.GetSize()[::-1], dtype=np.uint8)
    
    for segment_number in result.available_segments:
        segmentation_image_data = result.segment_data(segment_number)
        seg = np.swapaxes(segmentation_image_data, 0, -1)
        full_mask = np.logical_or(full_mask, seg).astype(np.uint8)

    # Convert to SimpleITK image for cropping
    sitk_mask = sitk.GetImageFromArray(full_mask)
    sitk_mask.CopyInformation(original_volume)

    # Adjust crop coordinates for (z, y, x) order
    crop_start = [crop_coords[0], crop_coords[2], crop_coords[4]]
    crop_size = [crop_coords[1] - crop_coords[0],
                 crop_coords[3] - crop_coords[2],
                 crop_coords[5] - crop_coords[4]]

    # Ensure crop is within image bounds
    image_size = sitk_mask.GetSize()
    crop_start = [max(0, min(s, image_size[i] - 1)) for i, s in enumerate(crop_start)]
    crop_size = [min(s, image_size[i] - crop_start[i]) for i, s in enumerate(crop_size)]

    print(f"Image size: {image_size}")
    print(f"Crop start: {crop_start}")
    print(f"Crop size: {crop_size}")

    cropped_mask = sitk.Crop(sitk_mask, crop_start, crop_size)

    # Save the cropped mask as NIFTI
    cropped_mask_path = os.path.splitext(output_path)[0] + "_cropped_mask.nii.gz"
    sitk.WriteImage(cropped_mask, cropped_mask_path)
    print(f"Cropped mask saved as {cropped_mask_path}")

    print(f"Cropped mask size: {cropped_mask.GetSize()}")

# The main function remains the same

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