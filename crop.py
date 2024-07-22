import os
from monai.transforms import (
    LoadImage,
    SaveImage,
    SpatialPad,
    Resize,
)
import numpy as np

def pad_and_crop_segmentation(seg_image, reference_image, crop_coords):
    print("Start pad_and_crop_segmentation")
    
    # Pad the segmentation to match the reference image size
    padder = SpatialPad(spatial_size=reference_image.shape, mode='constant')
    padded_seg = padder(seg_image)
    
    print("padded_seg shape: ", padded_seg.shape)
    print("original seg shape: ", seg_image.shape)
    
    # Crop the padded segmentation
    cropper = Resize(spatial_size=crop_coords[1::2])
    cropped_seg = cropper(padded_seg[None, ...])[0]
    
    print("End pad_and_crop_segmentation")
    return padded_seg, cropped_seg

def process_volume(mask_file, output_path, crop_coords, reference_dicom_folder):
    print(f"Mask file: {mask_file}")
    print(f"Reference DICOM folder: {reference_dicom_folder}")
    
    # Load the segmentation mask
    loader = LoadImage(image_only=True)
    seg_image = loader(mask_file)

    # Load the reference DICOM image
    reference_image = loader(reference_dicom_folder)

    # Pad and crop the segmentation
    padded_seg, cropped_seg = pad_and_crop_segmentation(seg_image, reference_image, crop_coords)

    # Save the full padded mask
    full_mask_path = os.path.splitext(output_path)[0] + "_full.nii.gz"
    saver = SaveImage(output_dir=os.path.dirname(full_mask_path), output_postfix="", output_ext=".nii.gz", resample=False)
    saver(padded_seg)
    print(f"Full mask saved as {full_mask_path}")

    # Save the cropped mask
    cropped_mask_path = os.path.splitext(output_path)[0] + "_cropped.nii.gz"
    saver = SaveImage(output_dir=os.path.dirname(cropped_mask_path), output_postfix="", output_ext=".nii.gz", resample=False)
    saver(cropped_seg)
    print(f"Cropped mask saved as {cropped_mask_path}")

    print(f"Cropped image size: {cropped_seg.shape}")
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

    # Process the volume
    process_volume(mask_file, output_path, crop_coords, reference_dicom_folder)

if __name__ == "__main__":
    main()