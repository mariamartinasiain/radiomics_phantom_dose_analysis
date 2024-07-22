import pydicom
import pydicom_seg
import numpy as np
import os

def pad_segmentation(seg, target_shape, start_index, end_index):
    print("Start pad_segmentation")
    padded_seg = np.zeros(target_shape, dtype=seg.dtype)
    print("padded_seg shape: ", padded_seg.shape)
    print("seg shape: ", seg.shape)
    print("start_index: ", start_index)
    print("end_index: ", end_index)
    padded_seg[:, :, start_index:end_index] = seg
    print("End pad_segmentation")
    return padded_seg

def find_closest_index(array, value):
    if not array:
        raise ValueError("Input array is empty")
    return np.argmin(np.abs(np.array(array) - value))

def crop_volume(mask_file, output_path, crop_coords, reference_dicom_folder):
    print(f"Mask file: {mask_file}")
    print(f"Reference DICOM folder: {reference_dicom_folder}")
    
    # Read the DICOM segmentation file
    dicom_seg = pydicom.dcmread(mask_file)
    reader = pydicom_seg.SegmentReader()
    result = reader.read(dicom_seg)

    # Read reference DICOM files
    dicom_files = [os.path.join(reference_dicom_folder, f) for f in os.listdir(reference_dicom_folder) if f.endswith('.dcm')]
    print(f"Number of potential DICOM files found: {len(dicom_files)}")
    dicom_files.sort()

    dicom_datasets = []
    all_instance_z_locations = []

    for f in dicom_files:
        try:
            ds = pydicom.dcmread(f)
            dicom_datasets.append(ds)
            if hasattr(ds, 'ImagePositionPatient'):
                all_instance_z_locations.append(float(ds.ImagePositionPatient[-1]))
            else:
                print(f"Warning: ImagePositionPatient not found in file {f}")
        except Exception as e:
            print(f"Error reading file {f}: {e}")

    print(f"Number of valid DICOM datasets: {len(dicom_datasets)}")
    print(f"Number of Z locations found: {len(all_instance_z_locations)}")

    if not all_instance_z_locations:
        raise ValueError("No valid Z locations found in DICOM files")

    all_referenced_z_locations = [
        float(f.PlanePositionSequence[0].ImagePositionPatient[-1])
        for f in dicom_seg.PerFrameFunctionalGroupsSequence
    ]
    all_referenced_z_locations = np.unique(all_referenced_z_locations)
    print(f"Number of referenced Z locations: {len(all_referenced_z_locations)}")
    print(f"Referenced Z locations: {all_referenced_z_locations}")

    min_referenced_z_location = min(all_referenced_z_locations)
    print(f"Minimum referenced Z location: {min_referenced_z_location}")

    # Find the closest index instead of exact match
    starting_index_global = find_closest_index(all_instance_z_locations, min_referenced_z_location)
    ending_index_global = starting_index_global + len(all_referenced_z_locations)

    print(f"Starting index: {starting_index_global}, Ending index: {ending_index_global}")

    # Process all segments into a single mask
    full_mask = np.zeros((512, 512, len(dicom_datasets)), dtype=np.uint8)
    
    # Extract segmentation data
    segmentation_data = result.segmentation_data[0]  # Assuming single segment

    # Change axes to match DICOM
    seg = np.fliplr(np.swapaxes(segmentation_data, 0, -1))

    # Pad segmentation to match DICOM dimensions
    padded_seg = pad_segmentation(
        seg, (512, 512, 512), starting_index_global, ending_index_global
    )

    # Crop the image
    crop_start = [crop_coords[0], crop_coords[2], crop_coords[4]]
    crop_size = [crop_coords[1] - crop_coords[0],
                 crop_coords[3] - crop_coords[2],
                 crop_coords[5] - crop_coords[4]]

    # Ensure crop is within image bounds
    image_size = padded_seg.shape
    crop_start = [max(0, min(s, image_size[i] - 1)) for i, s in enumerate(crop_start)]
    crop_size = [min(s, image_size[i] - crop_start[i]) for i, s in enumerate(crop_size)]

    print(f"Image size: {image_size}")
    print(f"Crop start: {crop_start}")
    print(f"Crop size: {crop_size}")

    cropped_image = padded_seg[crop_start[0]:crop_start[0]+crop_size[0],
                               crop_start[1]:crop_start[1]+crop_size[1],
                               crop_start[2]:crop_start[2]+crop_size[2]]

    print(f"Cropped image size: {cropped_image.shape}")

    # Save the cropped image as a new DICOM file
    new_dicom = pydicom.Dataset()
    new_dicom.file_meta = pydicom.Dataset()
    new_dicom.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    new_dicom.is_little_endian = True
    new_dicom.is_implicit_VR = False

    new_dicom.SOPClassUID = dicom_seg.SOPClassUID
    new_dicom.SOPInstanceUID = pydicom.uid.generate_uid()
    new_dicom.StudyInstanceUID = dicom_seg.StudyInstanceUID
    new_dicom.SeriesInstanceUID = pydicom.uid.generate_uid()
    new_dicom.FrameOfReferenceUID = dicom_seg.FrameOfReferenceUID

    new_dicom.Modality = 'SEG'
    new_dicom.SeriesDescription = 'Cropped Segmentation'

    # Add the pixel data
    new_dicom.PixelData = cropped_image.tobytes()
    new_dicom.Rows, new_dicom.Columns = cropped_image.shape[:2]
    new_dicom.SamplesPerPixel = 1
    new_dicom.PhotometricInterpretation = "MONOCHROME2"
    new_dicom.PixelRepresentation = 0
    new_dicom.BitsAllocated = 8
    new_dicom.BitsStored = 8
    new_dicom.HighBit = 7

    new_dicom.save_as(output_path)
    print(f"Cropped mask saved as {output_path}")

# The main function remains the same

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