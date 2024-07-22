import SimpleITK as sitk
import os

def crop_volume(input_path, output_path, crop_coords):
    # Read the image
    reader = sitk.ImageFileReader()
    reader.SetFileName(input_path)
    image = reader.Execute()

    # Crop the image
    size = [crop_coords[5] - crop_coords[4], 
            crop_coords[3] - crop_coords[2], 
            crop_coords[1] - crop_coords[0]]
    cropped_image = sitk.Crop(image, crop_coords[::2], size)

    # Write the cropped image
    writer = sitk.ImageFileWriter()
    writer.SetFileName(output_path)
    writer.Execute(cropped_image)

def main():
    base_path = "/mnt/nas4/datasets/ToCurate/QA4IQI/FinalDataset-TCIA-MultiCentric/Upl/A1"
    reference_volume = "A1_174008_691000_SOMATOM_Definition_Edge_ID23_Harmonized_10mGy_IR_NrFiles_343"
    mask_file = os.path.join(base_path, reference_volume, "mask", f"{reference_volume}.dcm")

    # Crop coordinates [z_start, z_end, y_start, y_end, x_start, x_end]
    crop_coords = [13, 323, 120, 395, 64, 445]

    # Output path for the cropped mask
    output_path = os.path.join(base_path, f"{reference_volume}_cropped_mask.dcm")

    # Crop the mask
    crop_volume(mask_file, output_path, crop_coords)

    print(f"Mask cropped and saved as {output_path}")

if __name__ == "__main__":
    main()