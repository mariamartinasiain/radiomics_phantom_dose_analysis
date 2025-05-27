import os
import numpy as np
import torch
import nibabel as nib
import csv
from tqdm import tqdm
from collections import defaultdict
from monai.transforms import Compose, LoadImage, EnsureType, Orientation, ScaleIntensityRange, CropForeground
from lighter_zoo import SegResEncoder
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

patch_size = np.array([32, 32, 16])  # Patch dimensions

# Pre-trained feature extraction model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SegResEncoder.from_pretrained("project-lighter/ct_fm_feature_extractor").to(device)
model.eval()

# Preprocessing pipeline
preprocess = Compose([
    LoadImage(ensure_channel_first=True),  # Load image and ensure channel dimension
    EnsureType(),                         # Ensure correct data type
    #Orientation(axcodes="SPL"),           # Standardize orientation
    # Scale intensity to [0,1] range, clipping outliers
    ScaleIntensityRange(
        a_min=-1024,    # Min HU value
        a_max=2048,     # Max HU value
        b_min=0,        # Target min
        b_max=1,        # Target max
        clip=True       # Clip values outside range
    ),
    CropForeground()    # Remove background to reduce computation
])

# Dictionary to track scanner-dose-IR occurrences
scanner_count = defaultdict(int)

def extract_patch(volume, center, patch_size=(32, 32, 16)):
    """Extracts a 3D patch centered at 'center' with 'patch_size'."""

    center = np.array(center).astype(int)

    # Extract patch
    half_size = [size // 2 for size in patch_size]
    
    print(center)

    # Create slices for each dimension
    slices = tuple(
        slice(max(0, center[i] - half_size[i]), min(volume.shape[i+1], center[i] + half_size[i])) 
        for i in range(3)
    )

    # Extract the patch
    patch = volume[:, slices[0], slices[1], slices[2]]

    # If the patch dimensions are smaller than expected (due to the image boundary), 
    # adjust by padding to match the desired size
    pad_depth = patch_size[0] - patch.shape[1]
    pad_height = patch_size[1] - patch.shape[2]
    pad_width = patch_size[2] - patch.shape[3]

    if pad_depth > 0 or pad_height > 0 or pad_width > 0:
        patch = torch.nn.functional.pad(patch, (0, pad_width, 0, pad_height, 0, pad_depth))

    print(patch.shape)

    return patch


def run_inference(nifti_dir, output_dir):
    nifti_files = [f for f in os.listdir(nifti_dir) if f.endswith(".nii")]
    print(nifti_files)
    os.makedirs(output_dir, exist_ok=True)
    patches_dir = "/mnt/nas7/data/maria/final_features/patches_lowdose"
    os.makedirs(patches_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, "features_ct-fm_low_dose_v2.csv")

    desired_spacing = [0.6641, 0.6641, 0.8]  # mínimo común

    with open(output_csv, "w", newline="") as csvfile:
        fieldnames = ["FileName", "ROI", "deepfeatures", "Manufacturer", "Dose"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for file in tqdm(nifti_files, desc="Processing CT scans"):
            nifti_path = os.path.join(nifti_dir, file)
            print(nifti_path)
            if not os.path.exists(nifti_path):
                print(f"File not found: {nifti_path}")
                continue

            # Paso 1: Preprocesar
            image = preprocess(nifti_path)  # shape: (1, H, W, D)

            # Paso 2: Obtener volumen sin canal
            volume = image[0]  # shape: (H, W, D)

            # Paso 3: Resamplear
            nifti_img = nib.load(nifti_path)
            affine = nifti_img.affine
            original_spacing = np.abs(np.diag(affine)[:3])
            zoom_factors = original_spacing / desired_spacing

            assert volume.ndim == 3, f"Expected 3D volume, got shape {volume.shape}"

            volume = np.swapaxes(volume, 0, 2)
            resampled_volume = zoom(volume, zoom=zoom_factors, order=3)  # still shape: (H', W', D')

            # Paso 4: Volver a añadir canal
            resampled_image = np.expand_dims(resampled_volume, axis=0)  # shape: (1, H', W', D')


            # Detect Dose type from filename
            if "FD_" in file:
                dose_type = "Full Dose"
            elif "QD_" in file:
                dose_type = "Quarter Dose"
            else:
                dose_type = "Unknown"

            file_base = file.replace(".nii.gz", "")

            for entry in roi_data:
                patient_id, size, center, roi_label, roi_id = entry
                if patient_id in file:
                    # Convert center to resampled coordinates
                    center_phys = np.array(center) * original_spacing
                    new_center = (center_phys / desired_spacing).astype(int)

                    patch = extract_patch(resampled_image, new_center)

                    # Save patch
                    patch2 = patch.squeeze(0)
                    print('patch2', patch2.shape)
                    patch_nifti = nib.Nifti1Image(patch2.astype(np.float32), affine=np.eye(4))
                    patch_name = f"{file_base}_{roi_id}_{roi_label}_v2.nii.gz"
                    nib.save(patch_nifti, os.path.join(patches_dir, patch_name))

                    # Extract features
                    with torch.no_grad():
                        patch = torch.from_numpy(patch).float().to('cuda')
                        #patch = patch.to('cuda')    # Move the tensor to the default GPU
                        print('patch', patch.shape)
                        output = model(patch.unsqueeze(0))[-1]
                        feature_vector = torch.nn.functional.adaptive_avg_pool3d(output, 1).squeeze()
                        feature_vector = feature_vector.cpu().numpy()  # Convert tensor to numpy array
                        formatted_feature_vector = "[" + ", ".join([str(x) for x in feature_vector]) + "]"

                    writer.writerow({
                        "FileName": file_base,
                        "ROI": roi_label,
                        "deepfeatures": formatted_feature_vector,
                        "Manufacturer": "Siemens",  # update if needed
                        "Dose": dose_type
                    })

    print("✅ Feature extraction and patch saving completed!")


if __name__ == "__main__":
    nifti_dir = "/mnt/nas7/data/maria/final_features/nifti_new_dataset"
    output_dir = "/mnt/nas7/data/maria/final_features/ct-fm_low_dose"
    os.makedirs(output_dir, exist_ok=True)
   
    run_inference(nifti_dir, output_dir)

