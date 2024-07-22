import SimpleITK as sitk
import pydicom
import os

def print_sitk_metadata(image):
    print("SimpleITK Metadata:")
    for key in image.GetMetaDataKeys():
        print(f"{key}: {image.GetMetaData(key)}")

def crop_volume(input_path, output_path, crop_coords):
    # Read the image
    reader = sitk.ImageFileReader()
    reader.SetFileName(input_path)
    reader.LoadPrivateTagsOn()
    image = reader.Execute()

    # Get image properties
    size = image.GetSize()
    spacing = image.GetSpacing()
    origin = image.GetOrigin()
    direction = image.GetDirection()

    print(f"\nOriginal Image properties:")
    print(f"Size: {size}")
    print(f"Spacing: {spacing}")
    print(f"Origin: {origin}")
    print(f"Direction: {direction}")

    # Calculate the shift based on the origin
    shift = np.array([-int(origin[0]), -int(origin[1]), -int(origin[2])])

    # Calculate new size to encompass the entire volume
    new_size = [
        max(size[0], crop_coords[5] + shift[0]),
        max(size[1], crop_coords[3] + shift[1]),
        max(size[2], crop_coords[1] + shift[2])
    ]

    # Create a new image with the extended size
    extended_image = sitk.Image(new_size, image.GetPixelID())
    extended_image.CopyInformation(image)
    extended_image.SetOrigin((0, 0, 0))  # Reset origin for simplicity

    # Paste the original image into the extended image
    paster = sitk.PasteImageFilter()
    paster.SetDestinationIndex((shift[0], shift[1], shift[2]))
    extended_image = paster.Execute(extended_image, image)

    print(f"\nExtended Image Size: {extended_image.GetSize()}")

    # Adjust crop coordinates based on the shift
    adjusted_coords = [
        crop_coords[4] + shift[0],  # x_start
        crop_coords[5] + shift[0],  # x_end
        crop_coords[2] + shift[1],  # y_start
        crop_coords[3] + shift[1],  # y_end
        crop_coords[0] + shift[2],  # z_start
        crop_coords[1] + shift[2]   # z_end
    ]

    # Calculate new size for cropping
    crop_size = [
        adjusted_coords[1] - adjusted_coords[0],
        adjusted_coords[3] - adjusted_coords[2],
        adjusted_coords[5] - adjusted_coords[4]
    ]

    print(f"Adjusted crop coordinates: {adjusted_coords}")
    print(f"Crop size: {crop_size}")

    # Crop the extended image
    cropped_image = sitk.Crop(extended_image, adjusted_coords[::2], crop_size)

    # Write the cropped image
    writer = sitk.ImageFileWriter()
    writer.SetFileName(output_path)
    writer.Execute(cropped_image)

    print(f"\nFinal cropped image size: {cropped_image.GetSize()}")

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

    print(f"\nMask cropped and saved as {output_path}")

if __name__ == "__main__":
    main()