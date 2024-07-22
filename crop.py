import SimpleITK as sitk
import os

def print_sitk_metadata(image):
    print("SimpleITK Metadata:")
    for key in image.GetMetaDataKeys():
        print(f"{key}: {image.GetMetaData(key)}")

def crop_volume(input_path, output_path, crop_coords):
    # Read the image
    if input_path.lower().endswith('.nii'):
        image = sitk.ReadImage(input_path)
    else:  # Assume DICOM
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(os.path.dirname(input_path))
        reader.SetFileNames(dicom_names)
        image = reader.Execute()

    # Print metadata
    print_sitk_metadata(image)

    # Get image size
    size = image.GetSize()
    spacing = image.GetSpacing()
    origin = image.GetOrigin()
    direction = image.GetDirection()

    print(f"\nImage properties:")
    print(f"Size: {size}")
    print(f"Spacing: {spacing}")
    print(f"Origin: {origin}")
    print(f"Direction: {direction}")

    # Flip crop coordinates from [z_start, z_end, y_start, y_end, x_start, x_end]
    # to [x_start, x_end, y_start, y_end, z_start, z_end]
    flipped_coords = crop_coords[4:6] + crop_coords[2:4] + crop_coords[0:2]

    # Calculate new size
    new_size = [flipped_coords[1] - flipped_coords[0],
                flipped_coords[3] - flipped_coords[2],
                flipped_coords[5] - flipped_coords[4]]

    print(f"\nCrop coordinates: {flipped_coords}")
    print(f"New size: {new_size}")

    # Crop the image
    cropped_image = sitk.Crop(image, flipped_coords[::2], new_size)

    # Write the cropped image
    writer = sitk.ImageFileWriter()
    writer.SetFileName(output_path)
    writer.Execute(cropped_image)

def main():
    base_path = "/mnt/nas4/datasets/ToCurate/QA4IQI/FinalDataset-TCIA-MultiCentric/Upl/A1"
    reference_volume = "A1_174008_691000_SOMATOM_Definition_Edge_ID23_Harmonized_10mGy_IR_NrFiles_343"
    mask_file = "/mnt/nas7/data/phantom_rois/uncompressed_rois2/segmentation_A1_174008_691000_SOMATOM_Definition_Edge_ID23_Harmonized_10mGy_IR_NrFiles_343-1-normal1.nii"

    # Crop coordinates [z_start, z_end, y_start, y_end, x_start, x_end]
    crop_coords = [13, 323, 120, 395, 64, 445]

    # Output path for the cropped mask
    output_path = os.path.join(base_path, f"{reference_volume}_cropped_mask.nii")

    # Crop the mask
    crop_volume(mask_file, output_path, crop_coords)

    print(f"\nMask cropped and saved as {output_path}")

if __name__ == "__main__":
    main()