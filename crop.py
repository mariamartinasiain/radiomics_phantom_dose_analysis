import os
import SimpleITK as sitk
import numpy as np

def pad_and_crop_segmentation(seg_image, reference_image, crop_coords):
    print("Start pad_and_crop_segmentation")
    print("Segmentation image size:", seg_image.GetSize())
    print("Reference image size:", reference_image.GetSize())
    
    # Pad the segmentation to match the reference image size
    padded_seg = sitk.ConstantPad(seg_image, 
                                  reference_image.GetSize() - seg_image.GetSize(),
                                  [0, 0, 0])
    
    print("Padded segmentation size:", padded_seg.GetSize())
    
    # Crop the padded segmentation
    crop_start = [crop_coords[4], crop_coords[2], crop_coords[0]]
    crop_size = [crop_coords[5] - crop_coords[4],
                 crop_coords[3] - crop_coords[2],
                 crop_coords[1] - crop_coords[0]]
    cropped_seg = sitk.Crop(padded_seg, crop_start, crop_size)
    
    print("Cropped segmentation size:", cropped_seg.GetSize())
    print("End pad_and_crop_segmentation")
    return padded_seg, cropped_seg

def process_volume(mask_file, output_path, crop_coords, reference_dicom_folder):
    print(f"Mask file: {mask_file}")
    print(f"Reference DICOM folder: {reference_dicom_folder}")
    
    # Load the segmentation mask
    seg_image = sitk.ReadImage(mask_file)
    print("Loaded segmentation mask size:", seg_image.GetSize())

    # Load the reference DICOM image
    reference_image = sitk.ReadImage(reference_dicom_folder)
    print("Loaded reference image size:", reference_image.GetSize())

    # Pad and crop the segmentation
    padded_seg, cropped_seg = pad_and_crop_segmentation(seg_image, reference_image, crop_coords)

    # Save the full padded mask
    full_mask_path = os.path.splitext(output_path)[0] + "_full.nii.gz"
    sitk.WriteImage(padded_seg, full_mask_path)
    print(f"Full mask saved as {full_mask_path}")

    # Save the cropped mask
    cropped_mask_path = os.path.splitext(output_path)[0] + "_cropped.nii.gz"
    sitk.WriteImage(cropped_seg, cropped_mask_path)
    print(f"Cropped mask saved as {cropped_mask_path}")

    print(f"Cropped image size: {cropped_seg.GetSize()}")
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