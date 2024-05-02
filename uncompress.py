import os
import nibabel as nib
from tqdm import tqdm

source_directory = 'image' # Update this path to your directory

parent_directory = os.path.dirname(source_directory)
target_directory = os.path.join(parent_directory, 'uncompressed_images2')
os.makedirs(target_directory, exist_ok=True)  

for filename in tqdm(os.listdir(source_directory)):
    if filename.endswith('.nii.gz'):
        file_path = os.path.join(source_directory, filename)
        img = nib.load(file_path)
        output_file_path = os.path.join(target_directory, filename[:-3])  # Removes '.gz'
        nib.save(img, output_file_path)

print("Decompression complete. Uncompressed files are saved in:", target_directory)
