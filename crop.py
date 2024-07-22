import SimpleITK as sitk
import pydicom
import os
import numpy as np

def print_sitk_metadata(image):
    print("SimpleITK Metadata:")
    for key in image.GetMetaDataKeys():
        print(f"{key}: {image.GetMetaData(key)}")

import numpy as np
import SimpleITK as sitk

import numpy as np
import SimpleITK as sitk

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
    shift = np.array([-int(round(origin[0])), -int(round(origin[1])), -int(round(origin[2]))])

    # Calculate new size to encompass the entire volume
    new_size = [
        int(max(size[0], crop_coords[5] + shift[0])),
        int(max(size[1], crop_coords[3] + shift[1])),
        int(max(size[2], crop_coords[1] + shift[2]))
    ]

    print(f"\nNew extended size: {new_size}")

    # Create a new image with the extended size
    extended_image = sitk.Image(new_size, image.GetPixelID())
    
    # Set the metadata for the extended image
    extended_image.SetSpacing(spacing)
    extended_image.SetDirection(direction)
    extended_image.SetOrigin((0, 0, 0))  # Reset origin for simplicity

    # Paste the original image into the extended image
    paster = sitk.PasteImageFilter()
    paster.SetDestinationIndex((int(shift[0]), int(shift[1]), int(shift[2])))
    extended_image = paster.Execute(extended_image, image)

    print(f"Extended Image Size: {extended_image.GetSize()}")

    # Adjust crop coordinates based on the shift
    adjusted_coords = [
        max(0, min(int(crop_coords[4] + shift[0]), new_size[0] - 1)),  # x_start
        max(0, min(int(crop_coords[5] + shift[0]), new_size[0])),      # x_end
        max(0, min(int(crop_coords[2] + shift[1]), new_size[1] - 1)),  # y_start
        max(0, min(int(crop_coords[3] + shift[1]), new_size[1])),      # y_end
        max(0, min(int(crop_coords[0] + shift[2]), new_size[2] - 1)),  # z_start
        max(0, min(int(crop_coords[1] + shift[2]), new_size[2]))       # z_end
    ]

    # Calculate new size for cropping
    crop_size = [
        max(1, adjusted_coords[1] - adjusted_coords[0]),
        max(1, adjusted_coords[3] - adjusted_coords[2]),
        max(1, adjusted_coords[5] - adjusted_coords[4])
    ]

    print(f"Adjusted crop coordinates: {adjusted_coords}")
    print(f"Crop size: {crop_size}")

    # Crop the extended image
    cropped_image = sitk.Crop(extended_image, adjusted_coords[::2], crop_size)

    # Set the origin of the cropped image
    new_origin = (
        origin[0] + adjusted_coords[0] * spacing[0],
        origin[1] + adjusted_coords[2] * spacing[1],
        origin[2] + adjusted_coords[4] * spacing[2]
    )
    cropped_image.SetOrigin(new_origin)

    # Write the cropped image
    writer = sitk.ImageFileWriter()
    writer.SetFileName(output_path)
    writer.Execute(cropped_image)

    print(f"\nFinal cropped image size: {cropped_image.GetSize()}")
    print(f"Final cropped image origin: {cropped_image.GetOrigin()}")
    print(f"Final cropped image spacing: {cropped_image.GetSpacing()}")
    print(f"Final cropped image direction: {cropped_image.GetDirection()}")

    return cropped_image

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