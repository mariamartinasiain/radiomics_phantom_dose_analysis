import logging
import os
from glob import glob
import json
import numpy as np
import nibabel as nib
import pydicom
import pydicom_seg
import logging
import SimpleITK as sitk

import sys
sys.path.append('/home/maria/radiomics_phantom_copy/')

logging.basicConfig(filename="std.log", 
					format='%(asctime)s %(message)s', 
					filemode='w') 


logger = logging.getLogger()


def pad_segmentation(segmentation, reference_image_shape, starting_index, ending_index):
    padded_seg = np.zeros(reference_image_shape, dtype=np.uint8)
    padded_seg[:, :, starting_index:ending_index] = segmentation
    return padded_seg


# Function to resample segmentation mask to match reference image spacing
def resample_segmentation(segmentation_image, reference_image, z_spacing):
    # Get the size of the reference image
    reference_size = segmentation_image.shape
    
    # Get the spacing from the reference image (this should be a tuple/list of 3 elements)
    #reference_spacing = reference_image.header.get_zooms()  # This will return a tuple of 3 values
    reference_spacing = [0.68359375, 0.68359375, 1.0]
    print(reference_spacing)

    # Resample the segmentation image to match the reference image size and spacing
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(reference_size)  # Set the size of the output image
    resampler.SetOutputSpacing(reference_spacing)  # Set the output spacing as a tuple of 3 values

    # Convertir la imagen de Nifti (de nibabel) a SimpleITK Image
    segmentation_image = sitk.GetImageFromArray(segmentation_image)

    # Resample the segmentation
    resampled_segmentation = resampler.Execute(segmentation_image)

    return resampled_segmentation



def generate_nifti_masks(dicom_file, nii, output_folder, output_folder_rois, main_folder, subfolders):
    print("Generating NIfTI masks...")

    output_file_seg_prefix = 'roi'
    output_file_seg_suffix = '.nii.gz'

    # Read DICOM segmentation file
    dicom_seg = pydicom.dcmread(dicom_file)
    reader = pydicom_seg.SegmentReader()
    result = reader.read(dicom_seg)

    # Get index to label mapping from the segmentation
    segment_labels = [s.SegmentLabel for s in dicom_seg.SegmentSequence]

    # Define metadata key for ImagePositionPatient
    dicom_key = "0020|0032"  # DICOM tag for ImagePositionPatient
    pixel_spacing_key = "0028|0030"  # PixelSpacing (XY spacing)
    slice_thickness_key = "0018|0050"  # SliceThickness (Z spacing)

    # Storage for Z positions
    all_instance_z_locations = []
    xy_spacing = None
    z_spacing = None

    for subfolder in subfolders:
        subfolder_path = os.path.join(main_folder, subfolder)

        if not os.path.isdir(subfolder_path):
            print(f"Skipping {subfolder} (not a directory)")
            continue

        print(f"\nProcessing folder: {subfolder}")

        # Initialize reader
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(subfolder_path)

        if not series_ids:
            print(f"No DICOM series found in {subfolder}.")
            continue

        series_file_names = reader.GetGDCMSeriesFileNames(subfolder_path, series_ids[0])
        reader.SetFileNames(series_file_names)

        # Extract metadata
        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()

        for i, file in enumerate(series_file_names):
            reader.SetFileNames([file])  # Read one file at a time
            image = reader.Execute()

            # Extract ImagePositionPatient
            if reader.HasMetaDataKey(0, dicom_key):
                image_position = reader.GetMetaData(0, dicom_key)
                z_position = float(image_position.split("\\")[-1])  # Extract the Z value
                all_instance_z_locations.append(z_position)

                    # Extract PixelSpacing (XY spacing)
            if reader.HasMetaDataKey(0, pixel_spacing_key):
                xy_spacing = reader.GetMetaData(0, pixel_spacing_key)
                xy_spacing = tuple(map(float, xy_spacing.split("\\")))  # Convert to tuple of floats

            # Extract SliceThickness (Z spacing)
            if reader.HasMetaDataKey(0, slice_thickness_key):
                z_spacing = reader.GetMetaData(0, slice_thickness_key)
                z_spacing = float(z_spacing)
        
        # Sort Z locations (important for index lookup)
        all_instance_z_locations = sorted(all_instance_z_locations)

    # After processing all the files, print the voxel spacing
    #print(f"XY Spacing (PixelSpacing): {xy_spacing}")
    #print(f"Z Spacing (SliceThickness): {z_spacing}")


    all_referenced_z_locations = [
        float(f.PlanePositionSequence[0].ImagePositionPatient[-1])
        for f in dicom_seg.PerFrameFunctionalGroupsSequence
    ]
    all_referenced_z_locations = np.unique(all_referenced_z_locations)

    min_referenced_z_location = min(all_referenced_z_locations)

    starting_index_global = all_instance_z_locations.index(min_referenced_z_location)
    ending_index_global = starting_index_global + len(all_referenced_z_locations)

    '''
    print("\nAll instance Z locations (sorted from DICOM series):")
    print(all_instance_z_locations)

    print("\nAll referenced Z locations (from segmentation DICOM):")
    print(all_referenced_z_locations)

    print(f"Min referenced Z location: {min_referenced_z_location}")
    print(f"Starting index in reference: {starting_index_global}")
    print(f"Ending index in reference: {ending_index_global}")

    print(f"\nReference NIfTI image shape: {nii.shape}")
    print(f"Voxel spacing in NIfTI: {nii.header.get_zooms()}")
    print(f"Affine transformation of NIfTI:\n{nii.affine}")
    '''


    # Process each available ROI segment
    nifti_roi_paths = []
    for segment_number in result.available_segments:
        segmentation_image_data = result.segment_data(segment_number)

        #print("\nOriginal segmentation shape before transformation:", segmentation_image_data.shape)

        # Change axes to match DICOM format (flip/rotate if necessary)
        seg = np.fliplr(np.swapaxes(segmentation_image_data, 0, -1))

        #print("Segmentation shape after flipping/swapping axes:", seg.shape)

        # Pad segmentation to match the reference image size
        padded_seg = pad_segmentation(seg, nii.shape, starting_index_global, ending_index_global)

        '''
        # Resample the segmentation to match the reference image spacing
        resampled_seg = resample_segmentation(seg, nii, nii.header.get_zooms())
        
        print("Resampled segmentation shape:", resampled_seg.GetSize())

        resampled_seg = sitk.GetArrayFromImage(resampled_seg)
        resampled_seg = np.transpose(resampled_seg, (2, 1, 0))
        '''
        
        print(f"\nPadded segmentation shape: {padded_seg.shape}")
        #print(f"Expected shape from reference image: {nii.shape}")
        #print(f"Starting index used: {starting_index_global}")
        #print(f"Ending index used: {ending_index_global}")


        # Create NIfTI image for the segmentation mask
        padded_seg_image = nib.Nifti1Image(padded_seg, nii.affine, nii.header)

        # Generate path for each ROI
        unique_id = os.path.basename(dicom_file)

        if segment_labels[segment_number - 1] == 'metastatsis':
            segment_labels[segment_number - 1] = 'metastasis'
            
        nifti_roi_path = f"{output_folder_rois}/{output_file_seg_prefix}_{unique_id}-{segment_number}-{segment_labels[segment_number - 1]}{output_file_seg_suffix}"
        nifti_roi_paths.append(nifti_roi_path)

        # Save the NIfTI mask for each ROI
        padded_seg_image.to_filename(nifti_roi_path, dtype=np.uint8)

    # JSON update with ROI paths (optional)
    json_filename = os.path.join(output_folder, "dataset_info2.json")
    json_data = []

    if os.path.exists(json_filename):
        with open(json_filename, 'r') as json_file:
            json_data = json.load(json_file)

    # Add new entries to the JSON for each ROI
    for roi_path in nifti_roi_paths:
        new_data_entry = {
            "roi": roi_path,
            "roi_label": roi_path.split('-')[-1].replace(output_file_seg_suffix, '')
        }
        json_data.append(new_data_entry)

    # Save updated JSON data
    with open(json_filename, 'w') as json_file:
        json.dump(json_data, json_file)

    return nifti_roi_paths  # Return the list of generated masks if needed


if __name__ == "__main__":
    generate_nifti_masks()