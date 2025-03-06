import nibabel as nib
import os

# Path to your dataset
dataset_dir = "/mnt/nas7/data/reza/registered_dataset_all_doses"
dataset_dir = "/mnt/nas7/data/reza/registered_dataset"

# List all NIfTI files in the directory
nifti_files = [f for f in os.listdir(dataset_dir) if f.endswith('.nii') or f.endswith('.nii.gz')]

# Load the first 3 NIfTI files
for nifti_file in nifti_files[:5]:
    # Full path to the file
    file_path = os.path.join(dataset_dir, nifti_file)
    
    # Load the NIfTI file
    img = nib.load(file_path)
    
    # Get the image data
    img_data = img.get_fdata()
    
    # Get the range of values in the image
    min_value = img_data.min()
    max_value = img_data.max()
    
    print(f"File: {nifti_file}")
    print(f"Value range: {min_value} to {max_value}\n")

print(len(nifti_files))
