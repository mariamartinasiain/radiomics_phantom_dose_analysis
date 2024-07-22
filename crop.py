import pydicom
import pydicom_seg
import numpy as np
import SimpleITK as sitk
import os

def pad_segmentation(seg, target_shape, start_index, end_index):
    padded_seg = np.zeros(target_shape, dtype=seg.dtype)
    padded_seg[:, :, start_index:end_index] = seg
    return padded_seg

def crop_volume(mask_file, output_path, crop_coords, reference_dicom_folder):
    # Read the DICOM segmentation file
    dicom_seg = pydicom.dcmread(mask_file)
    reader = pydicom_seg.SegmentReader()
    result = reader.read(dicom_seg)

    # Read reference DICOM files
    dicom_files = [os.path.join(reference_dicom_folder, f) for f in os.listdir(reference_dicom_folder) if f.endswith('.dcm')]
    dicom_datasets = [pydicom.dcmread(f) for f in dicom_files]

    # Find smallest patient Z position to define starting index
    all_instance_z_locations = [float(ds.ImagePositionPatient[-1]) for ds in dicom_datasets]

    all_referenced_z_locations = [
        float(f.PlanePositionSequence[0].ImagePositionPatient[-1])
        for f in dicom_seg.PerFrameFunctionalGroupsSequence
    ]
    all_referenced_z_locations = np.unique(all_referenced_z_locations)

    min_referenced_z_location = min(all_referenced_z_locations)

    starting_index_global = all_instance_z_locations.index(min_referenced_z_location)
    ending_index_global = starting_index_global + len(all_referenced_z_locations)

    # Process all segments into a single mask
    full_mask = np.zeros((512, 512, len(dicom_datasets)), dtype=np.uint8)
    
    for segment_number in result.available_segments:
        segmentation_image_data = result.segment_data(segment_number)

        # Change axes to match DICOM
        seg = np.fliplr(np.swapaxes(segmentation_image_data, 0, -1))

        # Pad segmentation to match DICOM dimensions
        padded_seg = pad_segmentation(
            seg, full_mask.shape, starting_index_global, ending_index_global
        )

        # Add this segment to the full mask
        full_mask = np.logical_or(full_mask, padded_seg).astype(np.uint8)

    # Convert to SimpleITK image for cropping
    sitk_image = sitk.GetImageFromArray(full_mask)

    # Crop the image
    crop_start = [crop_coords[4], crop_coords[2], crop_coords[0]]
    crop_size = [crop_coords[5] - crop_coords[4],
                 crop_coords[3] - crop_coords[2],
                 crop_coords[1] - crop_coords[0]]

    cropped_image = sitk.Crop(sitk_image, crop_start, crop_size)

    # Write the cropped image
    writer = sitk.ImageFileWriter()
    writer.SetFileName(output_path)
    writer.Execute(cropped_image)

    print(f"Cropped image size: {cropped_image.GetSize()}")
    print(f"Mask cropped and saved as {output_path}")

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