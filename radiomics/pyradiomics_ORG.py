from nifti_masks_generation import generate_nifti_masks, generate_nifti_masks_10regions, generate_patch_masks
from extract_pyradiomics_features import extract_features, extract_features_10regions, extract_features_org
import csv
import numpy as np
from tqdm import tqdm
import os
import nibabel as nib
import pandas as pd
import umap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

output_dir = "/mnt/nas7/data/maria/final_features/pyradiomics_extraction/ct-org"
os.makedirs(output_dir, exist_ok=True)

output_dir_masks = "/mnt/nas7/data/maria/final_features/pyradiomics_extraction/ct-org/masks"
os.makedirs(output_dir_masks, exist_ok=True)

output_dir_masks2 = "/mnt/nas7/data/maria/final_features/pyradiomics_extraction/ct-org/masks2"
os.makedirs(output_dir_masks2, exist_ok=True)

masks_dir = "/mnt/nas7/data/maria/final_features/CT-ORG/masks"
volumes_dir = "/mnt/nas7/data/maria/final_features/CT-ORG/volumes"


# Organ label mapping
label_to_name = {
    1: "liver",
    2: "bladder",
    3: "lungs",
    4: "kidneys",
    5: "bone",
    6: "brain"
}


def pad_to_match(vol1, vol2, nii):

    affine = nii.affine  # Assume both have same affine

    d1, h1, w1 = vol1.shape
    d2, h2, w2 = vol2.shape

    pad_d1 = max(d2 - d1, 0)
    pad_h1 = max(h2 - h1, 0)
    pad_w1 = max(w2 - w1, 0)

    pad_d2 = max(d1 - d2, 0)
    pad_h2 = max(h1 - h2, 0)
    pad_w2 = max(w1 - w2, 0)

    def get_pad_tuple(pad_d, pad_h, pad_w):
        return (
            (pad_d // 2, pad_d - pad_d // 2),
            (pad_h // 2, pad_h - pad_h // 2),
            (pad_w // 2, pad_w - pad_w // 2)
        )

    pad1 = get_pad_tuple(pad_d1, pad_h1, pad_w1)
    pad2 = get_pad_tuple(pad_d2, pad_h2, pad_w2)

    vol1_padded = np.pad(vol1, pad1, mode='constant', constant_values=0)
    vol2_padded = np.pad(vol2, pad2, mode='constant', constant_values=0)

    return vol1_padded, vol2_padded, affine

all_features_df = []

output_file = os.path.join(output_dir, "pyradiomics_features_ct-org.csv")

nifti_files = [f for f in os.listdir(volumes_dir) if f.endswith(".nii.gz")]


for file in tqdm(nifti_files, desc="Processing CT scans"):
    print(f"Processing {file}")
    file_path = os.path.join(volumes_dir, file)
    image_nii = nib.load(file_path)
    image_data = image_nii.get_fdata()
    affine = image_nii.affine

    # Load corresponding mask
    mask_file = file.replace("volume", "labels")
    mask_path = os.path.join(masks_dir, mask_file)

    if not os.path.exists(mask_path):
        print(f"Mask file {mask_path} does not exist, skipping {file}")
        continue

    mask_nii = nib.load(mask_path)
    mask_np = mask_nii.get_fdata()
    mask_np = np.round(mask_np).astype(np.int32)

    np.unique(mask_np)
    print('Mask unique values:', np.unique(mask_np))

    for label, name in label_to_name.items():
        print(label, name)
        organ_mask = (mask_np == label)

        if organ_mask.sum() == 0:
            print(f"Organ {name} not present in {file}, skipping")
            continue  # Skip if organ not present

        print('image:', image_data.shape, 'mask:', organ_mask.shape)

        if image_data.shape != organ_mask.shape:
            print('Different shape between image and mask')
            image_data, organ_mask, affine = pad_to_match(image_data, organ_mask, image_nii)
            print('After padding — image:', image_data.shape, 'mask:', organ_mask.shape)

        generate_patch_masks(
            image=image_data,
            mask=organ_mask,
            affine=affine,
            output_dir_masks=output_dir_masks,
            mask_filename=mask_file,
            name=name,
            patch_size=(64, 64, 32)
        )


def align_mask_to_image(image_nii, mask_nii, mask_path):
    image_shape = image_nii.shape
    mask_shape = mask_nii.shape
    image_affine = image_nii.affine
    mask_affine = mask_nii.affine

    print('Image shape:', image_shape, 'Mask shape:', mask_shape)
    print('Image affine:\n', image_affine)
    print('Mask affine:\n', mask_affine)

    # Check shape
    if image_shape != mask_shape:
        print(f"Padding or cropping mask {os.path.basename(mask_path)} from shape {mask_shape} to {image_shape}")
        # Pad or crop mask data to match image shape
        mask_data = mask_nii.get_fdata()
        
        # For simplicity, pad with zeros to match image shape
        padded_mask = np.zeros(image_shape, dtype=mask_data.dtype)

        # Determine min shape to copy
        min_shape = tuple(min(i_s, m_s) for i_s, m_s in zip(image_shape, mask_shape))

        slices_image = tuple(slice(0, ms) for ms in min_shape)
        slices_mask = tuple(slice(0, ms) for ms in min_shape)

        padded_mask[slices_image] = mask_data[slices_mask]

        # Use image affine to align spatial info exactly
        mask_nii_aligned = nib.Nifti1Image(padded_mask, image_affine)
        nib.save(mask_nii_aligned, mask_path)
        return mask_nii_aligned

    # Check spacing and orientation (rotation part of affine)
    # Compare direction cosines matrix (top-left 3x3 of affine)
    if not np.array_equal(image_affine[:3, 3], mask_affine[:3, 3]):
        print(f"Adjusting affine rotation/scale for mask {os.path.basename(mask_path)}")
        mask_data = mask_nii.get_fdata()
        fixed_affine = mask_affine.copy()
        fixed_affine[:3, 3] = np.array([0, 0, 0])
        fixed_affine[1, 1] *= -1
        mask_nii_aligned = nib.Nifti1Image(mask_data, fixed_affine)
        mask_path2 = os.path.join(output_dir_masks2, mask_file)
        nib.save(mask_nii_aligned, mask_path2)
        print(f"Mask {os.path.basename(mask_path2)} affine adjusted to match image affine.")
        affine = mask_nii_aligned.affine
        print('New mask affine:\n', affine)
        return mask_nii_aligned

    # Check origin translation
    if not np.allclose(image_affine[:3, 3], mask_affine[:3, 3], atol=1e-4):
        print(f"Adjusting origin translation for mask {os.path.basename(mask_path)}")
        mask_data = mask_nii.get_fdata()
        fixed_affine = mask_affine.copy()
        fixed_affine[:3, 3] = image_affine[:3, 3]
        mask_nii_aligned = nib.Nifti1Image(mask_data, fixed_affine)
        nib.save(mask_nii_aligned, mask_path)
        print(f"Mask {os.path.basename(mask_path)} origin adjusted to match image origin.")
        return mask_nii_aligned

    return mask_nii  # Already aligned


processed_pairs = set()
if os.path.exists(output_file):
    df_existing = pd.read_csv(output_file, usecols=["FileName", "ROI"])
    processed_pairs = set(zip(df_existing["FileName"].astype(str), df_existing["ROI"].astype(str)))

label_to_name_v2 = {
    1: "liver",
    2: "bladder",
    3: "lungs_right",
    4: "lungs_left",
    5: "kidneys_right",
    6: "kidneys_left",
    7: "bone",
    8: "brain"  
}

for file in tqdm(nifti_files, desc="Processing CT scans"):
    print(f"Processing {file}")
    file_path = os.path.join(volumes_dir, file)
    file_name = os.path.basename(file_path).replace(".nii.gz", "")
    print(f"File name: {file_name}")

    image_nii = nib.load(file_path)
    image_origin = image_nii.affine[:3, 3]

    mask_paths_for_file = []
    organ_names = []

    # Collect all mask paths for this image
    for label, name in label_to_name_v2.items():
        if (file_name, name) in processed_pairs:
            print('Skipping already processed pair:', file_name, name)
            continue

        mask_file = file.replace("volume", "labels").replace(".nii.gz", f"_{name}.nii.gz")
        mask_path = os.path.join(output_dir_masks, mask_file)

        if not os.path.exists(mask_path):
            print(f"Mask file {mask_path} does not exist, skipping organ: {name}")
            continue

        mask_nii = nib.load(mask_path)
        mask_nii = align_mask_to_image(image_nii, mask_nii, mask_path)

        mask_path2 = os.path.join(output_dir_masks2, mask_file)

        mask_paths_for_file.append(mask_path2)
        organ_names.append(name)

    if not mask_paths_for_file:
        print(f"No masks found for {file_name}, skipping feature extraction.")
        continue

    try:
        features_df = extract_features_org(file_path, mask_paths_for_file, organ_names)
    except ValueError as e:
        print(f"Geometry mismatch for {file} , skipping. Error: {e}")
        continue
    except Exception as e:
        print(f"Unexpected error for {file}: {e}")
        continue

    # Append to CSV once per image
    if not features_df.empty:
        file_exists = os.path.exists(output_file)
        features_df.to_csv(output_file, mode='a', header=not file_exists, index=False)
        print(f"Saved features from {file_name}")

print("Feature extraction complete!")



######## UMAP ########

csv_path = '/mnt/nas7/data/maria/final_features/pyradiomics_extraction/ct-org/pyradiomics_features_ct-org.csv'
df = pd.read_csv(csv_path)

# Delete non numeric columns
non_feature_columns = ['FileName', 'ROI']
feature_columns = [col for col in df.columns if col not in non_feature_columns]

# Extract radiomic features
X = df[feature_columns].values

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Group ROIs deleting 'left_' and 'right_'
df['ROI_grouped'] = df['ROI'].str.replace(r'_(left|right)$', '', regex=True)
grouped_labels = df['ROI_grouped'].values

# UMAP
reducer = umap.UMAP(n_neighbors=20, min_dist=0.4, n_components=2, random_state=24)
embedding = reducer.fit_transform(X_scaled)

unique_labels = np.unique(grouped_labels)

cmap = plt.get_cmap('tab10')

plt.figure(figsize=(10, 7))
for i, roi in enumerate(unique_labels):
    mask = grouped_labels == roi
    color = cmap(i % cmap.N)
    plt.scatter(embedding[mask, 0], embedding[mask, 1], label=roi, alpha=0.7, color=color)

plt.title("PyRadiomics features on CT-ORG data (UMAP)")
# plt.legend()  
plt.grid(True)

output_path = os.path.join(os.path.dirname(csv_path), 'pyradiomics_org_umap_final2.png')
plt.savefig(output_path)
plt.close()
