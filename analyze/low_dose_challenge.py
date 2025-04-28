import pydicom
import numpy as np
import os
import nibabel as nib


#### DICOM TO NIFTI

# Function to create the affine matrix from DICOM orientation
def get_affine(dicom_slices):
    orientation = dicom_slices[0].ImageOrientationPatient
    position = dicom_slices[0].ImagePositionPatient
    spacing_x, spacing_y = dicom_slices[0].PixelSpacing  # in-plane spacing
    spacing_z = float(dicom_slices[1].ImagePositionPatient[2] - dicom_slices[0].ImagePositionPatient[2])  # slice spacing
    
    x_dir = np.array(orientation[:3])
    y_dir = np.array(orientation[3:])
    z_dir = np.cross(x_dir, y_dir)
    
    # Apply voxel spacing to each direction vector
    x_dir = x_dir * spacing_x
    y_dir = y_dir * spacing_y
    z_dir = z_dir * spacing_z
    
    origin = np.array(position)
    
    affine = np.eye(4)
    affine[:3, 0] = x_dir
    affine[:3, 1] = y_dir
    affine[:3, 2] = z_dir
    affine[:3, 3] = origin
    
    print(f"Affine matrix (with spacing):\n{affine}")
    
    if np.any(np.isnan(affine)) or np.any(np.isinf(affine)):
        raise ValueError("Affine matrix contains invalid values (NaN or infinity)")
    
    lps_to_ras = np.diag([-1, -1, 1, 1])
    affine = lps_to_ras @ affine

    print(f"Affine matrix:\n{affine}")
    
    return affine


# Path to the main directory containing the subfolders (L067, L096, etc.)
main_dicom_dir = '/mnt/nas7/data/maria/final_features/QD_1mm/'

# Loop through each subfolder (e.g., L067, L096, etc.)
for subfolder in sorted(os.listdir(main_dicom_dir)):
    subfolder_path = os.path.join(main_dicom_dir, subfolder)
    
    # Check if the subfolder is indeed a directory
    if os.path.isdir(subfolder_path):
        # Now go into the 'full_1mm' directory inside each subfolder
        full_1mm_path = os.path.join(subfolder_path, 'quarter_1mm')
        
        # Check if 'full_1mm' exists
        if os.path.isdir(full_1mm_path):
            print(f"Processing folder: {full_1mm_path}")
            
            # Load all the .IMA files in the 'full_1mm' directory
            slices = []
            for filename in sorted(os.listdir(full_1mm_path)):
                if filename.lower().endswith('.ima'):
                    path = os.path.join(full_1mm_path, filename)
                    dicom = pydicom.dcmread(path)  # Read the DICOM file
                    slices.append(dicom)

            # Sort the slices by their Z position (ImagePositionPatient[2])
            slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

            # Create a 3D volume from the pixel data
            volume = np.stack([s.pixel_array for s in slices], axis=0)  # shape: (z, y, x)
            #volume = np.transpose(volume, (2, 1, 0))  # now shape: (x, y, z)

            # Convert to Hounsfield Units (HU)
            slope = float(slices[0].RescaleSlope)
            intercept = float(slices[0].RescaleIntercept)
            volume_hu = volume * slope + intercept

            # Print the shape of the resulting volume
            print(f"Volume shape for {full_1mm_path}: {volume_hu.shape}")

            # Get the correct affine matrix based on DICOM metadata
            affine = get_affine(slices)

            # Convert the volume to NIfTI format and save it
            nifti_image = nib.Nifti1Image(volume_hu, affine=affine)  # Use the correct affine matrix
            print(nifti_image.shape)
            
            # Define the output path for the NIfTI file
            output_nifti_path = os.path.join('/mnt/nas7/data/maria/final_features/nifti_new_dataset/', f'QD_{subfolder}.nii')

            # Save the NIfTI image
            nifti_image.to_filename(output_nifti_path)
            print(f"NIfTI image for {subfolder} saved to {output_nifti_path}")

            
#### PATCH EXTRACTION

from scipy.ndimage import zoom

roi_data = [
    ("L067", 15.2643, [125, 247, 501], "hemangioma", "L067_1"), # X Y Z
    ("L067", 13.0384, [114, 209, 358], "hemangioma", "L067_2"),
    ("L067", 17.0294, [242, 161, 527], "postop_defect", "L067_3"),
    ("L067", 4.0, [83, 276, 444], "benign_cyst", "L067_4"),
    ("L067", 5.65685, [188, 123, 444], "benign_cyst", "L067_5"),
    ("L096", 23.4094, [122, 192, 495], "metastasis", "L096_1"),
    ("L096", 14.0, [197, 219, 495], "metastasis", "L096_2"),
    ("L096", 8.94427, [85, 219, 475], "hemangioma", "L096_3"),
    ("L096", 7.0, [138, 243, 468], "metastasis", "L096_4"),
    ("L096", 11.4018, [207, 132, 437], "perfusion_defect", "L096_5"),
    ("L143", 16.1555, [169, 164, 299], "metastasis", "L143_1"),
    ("L143", 4.12311, [261, 168, 299], "benign_cyst", "L143_2"),
    ("L143", 15.2643, [182, 239, 264], "metastasis", "L143_3"),
    ("L143", 9.21954, [102, 200, 175], "metastasis", "L143_4"),
    ("L192", 5.65685, [221, 208, 518], "metastasis", "L192_1"),
    ("L192", 6.0, [200, 229, 484], "metastasis", "L192_2"),
    ("L192", 5.09902, [103, 336, 464], "metastasis", "L192_3"),
    ("L192", 10.4403, [120, 246, 407], "metastasis", "L192_4"),
    ("L192", 5.09902, [139, 321, 407], "metastasis", "L192_5"),
    ("L286", 7.28011, [177, 140, 493], "metastasis", "L286_1"),
    ("L291", 13.0384, [152, 271, 560], "metastasis", "L291_1"),
    ("L291", 16.2788, [164, 154, 560], "metastasis", "L291_2"),
    ("L291", 16.1245, [153, 273, 547], "metastasis", "L291_3"),
    ("L291", 20.6155, [150, 167, 483], "metastasis", "L291_4"),
    ("L291", 16.4012, [117, 274, 338], "metastasis", "L291_5"),
    ("L291", 13.0384, [169, 196, 490], "focal_fat", "L291_6"),
    ("L291", 14.2127, [133, 197, 444], "focal_fat", "L291_7"),
    ("L291", 11.4018, [194, 174, 483], "focal_fat", "L291_8"),
    ("L310", 13.1529, [150, 311, 467], "metastasis", "L310_1"),
    ("L506", 21.6333, [144, 264, 440], "metastasis", "L506_1"),
    ("L506", 12.083, [166, 218, 418], "metastasis", "L506_2"),
]


input_dir = "/mnt/nas7/data/maria/final_features/nifti_new_dataset"
patch_size = np.array([64, 64, 32])  # (X, Y, Z)
#patch_size = np.array([32, 64, 64])  # (Z, Y, X)

#desired_spacing = [0.8, 0.8, 0.8]  # mm
desired_spacing = [0.6641, 0.6641, 0.8] # mínimo común


######## FULL DOSE ########

for case_name, radius_mm, center, lesion, screenshot in roi_data:
    nifti_path = os.path.join(input_dir, f"FD_{case_name}.nii")
    if not os.path.exists(nifti_path):
        print(f"File not found: {nifti_path}")
        continue

    nifti_img = nib.load(nifti_path)
    volume = nifti_img.get_fdata()      # Z Y X

    affine = nifti_img.affine
    # Get original voxel spacing from the affine matrix
    original_spacing = np.abs(np.diag(affine)[:3])      # X Y Z 
    zoom_factors = original_spacing / desired_spacing

    print('zoom', zoom_factors)

    volume = np.swapaxes(volume, 0, 2)

    # Resample volume to desired voxel spacing
    resampled_volume = zoom(volume, zoom=zoom_factors, order=3)

    # Adjust the lesion center to resampled space
    center_phys = np.array(center) * original_spacing
    new_center = (center_phys / desired_spacing).astype(int)

    center = np.array(center).astype(int)

    print('original spacing', original_spacing)
    print('desired spacing', desired_spacing)

    print('center', center, 'new center', new_center)

    print('shape vol', volume.shape)
    print('shape new vol', resampled_volume.shape)
    

    z = center[2]
    z_resampled = new_center[2]
    print('z', z, 'new z', z_resampled)

    # Extract patch
    half_patch = patch_size // 2
    min_corner = np.maximum(new_center - half_patch, 0)
    max_corner = np.minimum(new_center + half_patch, resampled_volume.shape)
    print('shape', volume.shape)
    print('center', center)

    patch = resampled_volume[
        min_corner[0]:max_corner[0],
        min_corner[1]:max_corner[1],
        min_corner[2]:max_corner[2]
    ]
    
    print(patch.shape)

    output_dir = '/mnt/nas7/data/maria/final_features/patches_lowdose'
    patch_nifti = nib.Nifti1Image(patch, affine)
    patch_output_path = os.path.join(output_dir, f"FD_{screenshot}_{lesion}_patch_resampled.nii")
    nib.save(patch_nifti, patch_output_path)

    print(f"Saved patch: {patch_output_path}")


######## QUARTER DOSE ########

for case_name, radius_mm, center, lesion, screenshot in roi_data:
    nifti_path = os.path.join(input_dir, f"QD_{case_name}.nii")
    if not os.path.exists(nifti_path):
        print(f"File not found: {nifti_path}")
        continue

    nifti_img = nib.load(nifti_path)
    volume = nifti_img.get_fdata()      # Z Y X

    affine = nifti_img.affine
    # Get original voxel spacing from the affine matrix
    original_spacing = np.abs(np.diag(affine)[:3])      # X Y Z 
    zoom_factors = original_spacing / desired_spacing

    print('zoom', zoom_factors)

    volume = np.swapaxes(volume, 0, 2)

    # Resample volume to desired voxel spacing
    resampled_volume = zoom(volume, zoom=zoom_factors, order=3)

    # Adjust the lesion center to resampled space
    center_phys = np.array(center) * original_spacing
    new_center = (center_phys / desired_spacing).astype(int)


    center = np.array(center).astype(int)

    print('original spacing', original_spacing)
    print('desired spacing', desired_spacing)

    print('center', center, 'new center', new_center)

    print('shape vol', volume.shape)
    print('shape new vol', resampled_volume.shape)
    

    z = center[2]
    z_resampled = new_center[2]
    print('z', z, 'new z', z_resampled)

    # Extract patch
    half_patch = patch_size // 2
    min_corner = np.maximum(new_center - half_patch, 0)
    max_corner = np.minimum(new_center + half_patch, resampled_volume.shape)
    print('shape', volume.shape)
    print('center', center)

    patch = resampled_volume[
        min_corner[0]:max_corner[0],
        min_corner[1]:max_corner[1],
        min_corner[2]:max_corner[2]
    ]
    
    print(patch.shape)

    output_dir = '/mnt/nas7/data/maria/final_features/patches_lowdose'
    patch_nifti = nib.Nifti1Image(patch, affine)
    patch_output_path = os.path.join(output_dir, f"QD_{screenshot}_{lesion}_patch_resampled.nii")
    nib.save(patch_nifti, patch_output_path)

    print(f"Saved patch: {patch_output_path}")

