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

def nib_to_sitk(nib_img):
    data = nib_img.get_fdata()
    affine = nib_img.affine

    sitk_img = sitk.GetImageFromArray(data.astype(np.float32))
    spacing = np.abs(affine.diagonal())[:3]  # assumes affine is only scaling + translation
    origin = affine[:3, 3]
    direction = affine[:3, :3].flatten().tolist()

    sitk_img.SetSpacing(tuple(spacing))
    sitk_img.SetOrigin(tuple(origin))
    sitk_img.SetDirection(direction)
    return sitk_img

# Function to resample segmentation mask to match reference image spacing
def resample_segmentation(segmentation_array, reference_image):
    import numpy as np
    import SimpleITK as sitk
    import nibabel as nib

    # Convert numpy segmentation to SimpleITK image
    seg_sitk = sitk.GetImageFromArray(segmentation_array)

    # Convert Nibabel NIfTI to SimpleITK if needed
    if isinstance(reference_image, nib.Nifti1Image):
        ref_sitk = nib_to_sitk(reference_image)
    elif isinstance(reference_image, str):
        ref_sitk = sitk.ReadImage(reference_image)
    elif isinstance(reference_image, sitk.Image):
        ref_sitk = reference_image
    else:
        raise TypeError("Unsupported reference image type")

    # Set spacing/origin/direction on segmentation to match reference
    seg_sitk.SetSpacing(ref_sitk.GetSpacing())
    seg_sitk.SetOrigin(ref_sitk.GetOrigin())
    seg_sitk.SetDirection(ref_sitk.GetDirection())

    # Print for debugging
    print(f"Original segmentation values: {np.unique(sitk.GetArrayFromImage(seg_sitk))}")
    print(f"Reference spacing: {ref_sitk.GetSpacing()}")
    print(f"Reference origin: {ref_sitk.GetOrigin()}")
    print(f"Reference direction: {ref_sitk.GetDirection()}")

    # Resample
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref_sitk)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)

    resampled_seg = resampler.Execute(seg_sitk)
    resampled_seg_array = sitk.GetArrayFromImage(resampled_seg)

    print(f"Unique values in resampled segmentation: {np.unique(resampled_seg_array)}")
    print(f"Resampled spacing: {resampled_seg.GetSpacing()}")
    print(f"Resampled size: {resampled_seg.GetSize()}")

    return resampled_seg_array


def generate_nifti_masks(dicom_file, nii, output_folder, output_folder_rois, main_folder, subfolders):
    print("Generating NIfTI masks...")

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


    all_referenced_z_locations = [
        float(f.PlanePositionSequence[0].ImagePositionPatient[-1])
        for f in dicom_seg.PerFrameFunctionalGroupsSequence
    ]
    all_referenced_z_locations = np.unique(all_referenced_z_locations)

    min_referenced_z_location = min(all_referenced_z_locations)

    # Find closest match for the starting index
    starting_index_global = int(np.argmin(np.abs(np.array(all_instance_z_locations) - min_referenced_z_location)))
    print(starting_index_global)
    ending_index_global = starting_index_global + len(all_referenced_z_locations)

    # Process each available ROI segment
    nifti_roi_paths = []
    for segment_number in result.available_segments:
        segmentation_image_data = result.segment_data(segment_number)

        # Change axes to match DICOM format (flip/rotate if necessary)
        seg = np.fliplr(np.swapaxes(segmentation_image_data, 0, -1))

        # Pad segmentation to match the reference image size
        padded_seg = pad_segmentation(seg, nii.shape, starting_index_global, ending_index_global)

        # Print the original shape and spacing of the padded segmentation
        print(f"Original segmentation shape: {seg.shape}")
        print(f"Original padded segmentation shape: {padded_seg.shape}")

        # Resample the segmentation to match spacing of reference image
        resampled_seg = resample_segmentation(padded_seg, nii)
        print(f"Resampled segmentation shape: {resampled_seg.shape}")

        # Create NIfTI image for the segmentation mask
        resampled_seg_image = nib.Nifti1Image(resampled_seg.astype(np.uint8), nii.affine)

        print(f"Unique values in resampled segmentation array: {np.unique(resampled_seg)}")

        resampled_seg_data = resampled_seg_image.get_fdata()  # Get data from the NIfTI image object
        print(f"Unique values in resampled segmentation NIfTI image: {np.unique(resampled_seg_data)}")

        # Generate path for each ROI
        unique_id = os.path.basename(dicom_file)

        if segment_labels[segment_number - 1] == 'metastatsis':
            segment_labels[segment_number - 1] = 'metastasis'

        # Generate path for the ROI and save the resampled segmentation mask
        nifti_roi_path = f"{output_folder_rois}/{unique_id}_{segment_labels[segment_number - 1]}{output_file_seg_suffix}"
        resampled_seg_image.to_filename(nifti_roi_path)

        # Print the path where the NIfTI mask is saved
        print(f"Saved NIfTI mask at: {nifti_roi_path}")
            
        nifti_roi_paths.append(nifti_roi_path)

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
            "roi_label": roi_path.split('_')[-1].replace(output_file_seg_suffix, '')
        }
        json_data.append(new_data_entry)

    # Save updated JSON data
    with open(json_filename, 'w') as json_file:
        json.dump(json_data, json_file)

    return nifti_roi_paths  # Return the list of generated masks if needed


def flip_volume(volume, axis=0):
    volume = np.swapaxes(volume, 0, axis)
    volume_flipped = np.zeros_like(volume)
    nslices = volume.shape[0]
    for i in range(nslices):
         volume_flipped[i, ...] = volume[nslices - i - 1, ...]
    volume_flipped = np.swapaxes(volume_flipped, axis, 0)
    return volume_flipped


def generate_nifti_masks_10regions(nii_path, json_path, output_folder, patch_size=(32, 32, 16)):
    os.makedirs(output_folder, exist_ok=True)

    # Load the image
    nii_img = nib.load(nii_path)
    img_data = nii_img.get_fdata()
    affine = nii_img.affine
    print(img_data.shape)

    img_data = img_data.transpose(1, 0, 2)
    img_data = flip_volume(img_data, axis=0)
    img_shape = img_data.shape
 

    # Load the ROI centers from JSON
    with open(json_path, 'r') as f:
        roi_dict = json.load(f)

    saved_paths = []

    for roi_name, center in roi_dict.items():
 
        assert len(img_shape) == 3
        assert len(center) == 3
        assert len(patch_size) == 3

        half_size = [s // 2 for s in patch_size]

        # Calculate start and end coordinates, clipped to image bounds
        starts = [max(0, center[i] - half_size[i]) for i in range(3)]
        ends = [min(img_shape[i], center[i] + half_size[i]) for i in range(3)]

        # Create the binary mask
        mask = np.zeros(img_shape, dtype=np.uint8)
        z_start, y_start, x_start = starts
        z_end, y_end, x_end = ends
        mask[z_start:z_end, y_start:y_end, x_start:x_end] = 1

        mask = np.flip(mask, axis=0)
        mask = np.transpose(mask, (1, 0, 2))

        # Create NIfTI image
        mask_nifti = nib.Nifti1Image(mask, affine)

        # Save with appropriate name
        mask_path = os.path.join(output_folder, f"{roi_name}.nii.gz")
        nib.save(mask_nifti, mask_path)
        print(f"Saved mask: {mask_path}")

        saved_paths.append(mask_path)

    return saved_paths



def generate_patch_masks(image, mask, affine, output_dir_masks, mask_filename, name, patch_size=(64, 64, 32)):
    """
    Generate a binary patch mask around the center of a labeled region (or multiple parts like lungs/kidneys)
    and save it as a NIfTI file with the name as suffix.
    """
    W, H, D = image.shape
    pw, ph, pd = patch_size

    # Strip extension from the filename (labels-0.nii.gz → labels-0)
    base_filename = os.path.splitext(os.path.splitext(mask_filename)[0])[0]

    def create_mask(center_w, center_h, center_d, suffix):
        # Calculate start and end coordinates, ensuring patch stays within bounds
        w_start = max(center_w - pw // 2, 0)
        h_start = max(center_h - ph // 2, 0)
        d_start = max(center_d - pd // 2, 0)

        w_end = min(w_start + pw, W)
        h_end = min(h_start + ph, H)
        d_end = min(d_start + pd, D)

        w_start = max(w_end - pw, 0)
        h_start = max(h_end - ph, 0)
        d_start = max(d_end - pd, 0)

        # Create the patch mask
        patch_mask = np.zeros((W, H, D), dtype=np.uint8)
        patch_mask[w_start:w_end, h_start:h_end, d_start:d_end] = 1

        # Save the mask
        mask_nii = nib.Nifti1Image(patch_mask, affine=affine)
        nib.save(mask_nii, os.path.join(output_dir_masks, f"{base_filename}_{suffix}.nii.gz"))

    # Special case for paired organs
    if name in ['lungs', 'kidneys']:
        center = W // 2
        left_mask = np.copy(mask)
        right_mask = np.copy(mask)
        left_mask[center:, :, :] = 0
        right_mask[:center, :, :] = 0

        for side_mask, suffix in zip([left_mask, right_mask], [f"{name}_left", f"{name}_right"]):
            nonzero = np.argwhere(side_mask > 0)
            if nonzero.size == 0:
                continue
            center_coords = np.round(nonzero.mean(axis=0)).astype(int)
            create_mask(*center_coords, suffix)

    else:
        nonzero = np.argwhere(mask > 0)
        if nonzero.size == 0:
            create_mask(pw // 2, ph // 2, pd // 2, name)
        else:
            center_coords = np.round(nonzero.mean(axis=0)).astype(int)
            create_mask(*center_coords, name)



if __name__ == "__main__":
    generate_patch_masks()