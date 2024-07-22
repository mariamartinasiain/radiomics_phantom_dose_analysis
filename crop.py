import SimpleITK as sitk
import pydicom
import os

def check_mask_metadata(file_path):
    dcm = pydicom.dcmread(file_path)
    
    print(f"Image size: {dcm.Rows}x{dcm.Columns}")
    if hasattr(dcm, 'NumberOfFrames'):
        print(f"Number of frames: {dcm.NumberOfFrames}")
    
    print("\nAll DICOM tags:")
    for elem in dcm:
        if elem.VR != "SQ":  # Skip sequence items to avoid clutter
            print(f"{elem.tag}: {elem.name} = {repr(elem.value)}")

# The rest of the script remains the same

# The rest of the script remains the same

def crop_volume(input_path, output_path, crop_coords):
    # Check mask metadata
    print("Mask metadata:")
    check_mask_metadata(input_path)
    
    # Read the image
    reader = sitk.ImageFileReader()
    reader.SetFileName(input_path)
    image = reader.Execute()

    # Get image size
    size = image.GetSize()
    print(f"\nOriginal image size (from SimpleITK): {size}")

    # Flip crop coordinates from [z_start, z_end, y_start, y_end, x_start, x_end]
    # to [x_start, x_end, y_start, y_end, z_start, z_end]
    flipped_coords = crop_coords[4:6] + crop_coords[2:4] + crop_coords[0:2]
    
    # Calculate new size
    new_size = [flipped_coords[1] - flipped_coords[0], 
                flipped_coords[3] - flipped_coords[2], 
                flipped_coords[5] - flipped_coords[4]]
    
    print(f"Crop coordinates: {flipped_coords}")
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