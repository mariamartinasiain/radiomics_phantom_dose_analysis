import nibabel as nib
import os

input_folder = "/mnt/nas7/data/reza/registered_dataset_all_doses_pad_updated_NonUnifiedOrigins/"
output_folder = "/mnt/nas7/data/maria/final_features/registered_niftis_new/"
os.makedirs(output_folder, exist_ok=True)

# Reference image 
reference_image_path = os.path.join(input_folder, "A1_174008_691000_SOMATOM_Definition_Edge_ID23_Harmonized_10mGy_IR_NrFiles_343.nii.gz")

ref_nifti = nib.load(reference_image_path)
ref_affine = ref_nifti.affine.copy()  # Get the affine matrix of the reference

new_images_count = 0
skipped_images_count = 0

# Process all NIfTI files in the folder
for filename in sorted(os.listdir(input_folder)):
    if filename.endswith(".nii") or filename.endswith(".nii.gz"):
        img_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        # Check if the file already exists in the output folder
        if os.path.exists(output_path):
            print(f"Skipping {filename} as it already exists in the output folder.")
            skipped_images_count += 1 
            continue
        
        try:
            nifti = nib.load(img_path)
            print(f"Processing image: {filename}")
            
            # Create new NIfTI with the reference affine
            new_nifti = nib.Nifti1Image(nifti.get_fdata(), ref_affine, nifti.header)
            
            nib.save(new_nifti, output_path)
            new_images_count += 1 
        
        except FileNotFoundError:
            print(f"Warning: File {filename} not found, skipping...")
            continue

# Output the results
print(f"Total images processed: {new_images_count + skipped_images_count}")


