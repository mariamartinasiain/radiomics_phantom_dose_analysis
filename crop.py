import os
from monai.transforms import (
    LoadImage,
    SaveImage,
    Compose,
    SpatialPad,
    SpatialCrop
)

def create_transform_pipeline(reference_size, crop_coords):
    return Compose([
        LoadImage(ensure_channel_first=True),
        SpatialPad(spatial_size=(reference_size[0], reference_size[1], reference_size[2]), mode='constant'),
        SpatialCrop(roi_start=[crop_coords[4], crop_coords[2], crop_coords[0]],
                    roi_end=[crop_coords[5], crop_coords[3], crop_coords[1]])
    ])

def process_volume(mask_file, output_path, crop_coords, reference_dicom_folder):
    print(f"Mask file: {mask_file}")
    print(f"Reference DICOM folder: {reference_dicom_folder}")
    print(f"Output path: {output_path}")

    # Load reference image to get its size
    reference_loader = LoadImage(image_only=True)
    reference_image = reference_loader(reference_dicom_folder)
    reference_size = reference_image.shape

    mask_loader = LoadImage(image_only=True)
    mask = mask_loader(mask_file)
    print(f"Mask shape: {mask.shape}")
    print(f"Reference image shape: {reference_size}")

    # Create and apply transform pipeline
    transform = create_transform_pipeline(reference_size, crop_coords)
    processed_mask = transform(mask_file)

    # Save the processed mask
    saver = SaveImage(output_dir=os.path.dirname(output_path), 
                      output_postfix="",
                      output_ext=".nii.gz", 
                      resample=False,
                      squeeze_end_dims=True,
                      print_log=True)
    saver(processed_mask)

    print(f"Processed mask shape: {processed_mask.shape}")
    print(f"Mask processed and saved as {output_path}")

def main():
    base_path = "/mnt/nas4/datasets/ToCurate/QA4IQI/FinalDataset-TCIA-MultiCentric/Upl/A1"
    reference_volume = "A1_174008_691000_SOMATOM_Definition_Edge_ID23_Harmonized_10mGy_IR_NrFiles_343"
    mask_file = os.path.join(base_path, reference_volume, "mask", f"{reference_volume}.dcm")
    reference_dicom_folder = os.path.join(base_path, reference_volume)

    # Crop coordinates [z_start, z_end, y_start, y_end, x_start, x_end]
    crop_coords = [13, 323, 120, 395, 64, 445]

    # Create a new directory in the user's home folder
    output_dir = os.path.expanduser("~/processed_masks")
    os.makedirs(output_dir, exist_ok=True)

    # Output path for the processed mask
    output_path = os.path.join(output_dir, f"{reference_volume}_processed_mask.nii.gz")

    # Process the volume
    process_volume(mask_file, output_path, crop_coords, reference_dicom_folder)

    print(f"Processed mask saved to: {output_path}")

if __name__ == "__main__":
    main()
