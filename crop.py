import os
from monai.transforms import (
    LoadImage,
    SaveImage,
    Compose,
    SpatialPad,
    SpatialCrop
)

def create_transform_pipeline(reference_size, crop_coords, output_path):
    return Compose([
        LoadImage(image_only=True),
        SpatialPad(spatial_size=reference_size, mode='constant'),
        SpatialCrop(roi_start=[crop_coords[4], crop_coords[2], crop_coords[0]],
                    roi_end=[crop_coords[5], crop_coords[3], crop_coords[1]]),
        SaveImage(output_dir=os.path.dirname(output_path), output_postfix="", output_ext=".nii.gz", resample=False)
    ])

def process_volume(mask_file, output_path, crop_coords, reference_dicom_folder):
    print(f"Mask file: {mask_file}")
    print(f"Reference DICOM folder: {reference_dicom_folder}")

    # Load reference image to get its size
    reference_loader = LoadImage(image_only=True)
    reference_image = reference_loader(reference_dicom_folder)
    reference_size = reference_image.shape[1:]  # Assuming channel-first format

    # Create and apply transform pipeline
    transform = create_transform_pipeline(reference_size, crop_coords, output_path)
    processed_mask = transform(mask_file)
    
    print(f"Mask processed and saved as {output_path}")

def main():
    base_path = "/mnt/nas4/datasets/ToCurate/QA4IQI/FinalDataset-TCIA-MultiCentric/Upl/A1"
    reference_volume = "A1_174008_691000_SOMATOM_Definition_Edge_ID23_Harmonized_10mGy_IR_NrFiles_343"
    mask_file = os.path.join(base_path, reference_volume, "mask", f"{reference_volume}.dcm")
    reference_dicom_folder = os.path.join(base_path, reference_volume)

    # Crop coordinates [z_start, z_end, y_start, y_end, x_start, x_end]
    crop_coords = [13, 323, 120, 395, 64, 445]

    # Output path for the processed mask
    output_path = os.path.join(base_path, f"{reference_volume}_processed_mask.nii.gz")

    # Process the volume
    process_volume(mask_file, output_path, crop_coords, reference_dicom_folder)

if __name__ == "__main__":
    main()