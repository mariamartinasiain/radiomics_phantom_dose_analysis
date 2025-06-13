import os
import numpy as np
import torch
import nibabel as nib
import csv
from tqdm import tqdm
from monai.transforms import Compose, LoadImage, EnsureType, ScaleIntensityRange, CropForeground
from lighter_zoo import SegResEncoder
import torch.nn.functional as F
import pandas as pd
import umap
import matplotlib.pyplot as plt
import ast
from sklearn.preprocessing import StandardScaler

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained model
model = SegResEncoder.from_pretrained("project-lighter/ct_fm_feature_extractor").to(device)
model.eval()

# Preprocessing pipeline
preprocess = Compose([
    LoadImage(ensure_channel_first=True),
    EnsureType(),
    ScaleIntensityRange(a_min=-1024, a_max=2048, b_min=0, b_max=1, clip=True),
    CropForeground()
])

# Organ label mapping
label_to_name = {
    1: "liver",
    2: "bladder",
    3: "lungs",
    4: "kidneys",
    5: "bone",
    6: "brain"
}

def crop_patch_from_mask(
    tensor: torch.Tensor, 
    mask: torch.Tensor, 
    patch_size=(64, 64, 32),
    name: str = ''
) -> list:
    """
    Crops one or two 3D patches centered at the mask. If name indicates two organs (lungs/kidneys),
    splits the mask in half and returns one patch per half.

    Returns:
    - List of cropped tensors, each of shape [C, patch_size[0], patch_size[1], patch_size[2]]
    """
    assert tensor.dim() == 4 and mask.dim() == 4, "Expected tensors of shape [C, W, H, D]"
    _, W, H, D = tensor.shape
    pw, ph, pd = patch_size

    def crop_at(center_w, center_h, center_d):
        w_start = max(center_w - pw // 2, 0)
        h_start = max(center_h - ph // 2, 0)
        d_start = max(center_d - pd // 2, 0)

        w_end = min(w_start + pw, W)
        h_end = min(h_start + ph, H)
        d_end = min(d_start + pd, D)

        w_start = max(w_end - pw, 0)
        h_start = max(h_end - ph, 0)
        d_start = max(d_end - pd, 0)

        return tensor[:, w_start:w_end, h_start:h_end, d_start:d_end]

    patches = []

    if name in ['lungs', 'kidneys']:
        # Split mask left and right (based on width W)
        left_mask = mask.clone()
        left_mask[:, :W//2, :, :] = 0
        right_mask = mask.clone()
        right_mask[:, W//2:, :, :] = 0

        #patch_np = left_mask.squeeze(0).cpu().numpy()
        #patch_affine = np.eye(4)
        #patch_nii = nib.Nifti1Image(patch_np, affine=patch_affine)
        #patch_filename = os.path.join(output_dir, f"left_mask_{name}.nii.gz")
        #nib.save(patch_nii, patch_filename)

        for side_mask, suffix in zip([left_mask, right_mask], ['left', 'right']):
            nonzero = side_mask.sum(dim=0).nonzero(as_tuple=False)
            if nonzero.size(0) == 0:
                continue
            center_w, center_h, center_d = nonzero.float().mean(dim=0).round().long().tolist()
            patch = crop_at(center_w, center_h, center_d)
            patches.append((patch, f"{name}_{suffix}"))

    else:
        nonzero = mask.sum(dim=0).nonzero(as_tuple=False)
        if nonzero.size(0) == 0:
            # fallback crop from corner
            patch = tensor[:, :pw, :ph, :pd]
            patches.append((patch, name))
        else:
            center_w, center_h, center_d = nonzero.float().mean(dim=0).round().long().tolist()
            patch = crop_at(center_w, center_h, center_d)
            patches.append((patch, name))

    return patches


def pad_to_match(tensor1, tensor2):
    _, d1, h1, w1 = tensor1.shape
    _, d2, h2, w2 = tensor2.shape

    # Determine required padding for each tensor
    pad_d1 = max(d2 - d1, 0)
    pad_h1 = max(h2 - h1, 0)
    pad_w1 = max(w2 - w1, 0)

    pad_d2 = max(d1 - d2, 0)
    pad_h2 = max(h1 - h2, 0)
    pad_w2 = max(w1 - w2, 0)

    # Create padding tuples
    pad1 = [
        pad_w1 // 2, pad_w1 - pad_w1 // 2,
        pad_h1 // 2, pad_h1 - pad_h1 // 2,
        pad_d1 // 2, pad_d1 - pad_d1 // 2
    ]
    pad2 = [
        pad_w2 // 2, pad_w2 - pad_w2 // 2,
        pad_h2 // 2, pad_h2 - pad_h2 // 2,
        pad_d2 // 2, pad_d2 - pad_d2 // 2
    ]

    padded_tensor1 = F.pad(tensor1, pad1, mode='constant', value=0) if any(p > 0 for p in pad1) else tensor1
    padded_tensor2 = F.pad(tensor2, pad2, mode='constant', value=0) if any(p > 0 for p in pad2) else tensor2

    return padded_tensor1, padded_tensor2

def run_inference(nifti_dir, masks_dir, output_dir):
    nifti_files = [f for f in os.listdir(nifti_dir) if f.endswith(".nii.gz")]
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, "features_ct-fm_org_v3.csv")

    patches_dir = os.path.join(output_dir, "patches")
    os.makedirs(patches_dir, exist_ok=True)

    with open(output_csv, "w", newline="") as csvfile:
        fieldnames = ["FileName", "ROI", "deepfeatures"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for file in tqdm(nifti_files, desc="Processing CT scans"):
            print(f"Processing {file}")
            file_path = os.path.join(nifti_dir, file)
            image = preprocess(file_path)

            # Load corresponding mask
            mask_file = file.replace("volume", "labels")
            mask_path = os.path.join(masks_dir, mask_file)
            mask_nii = nib.load(mask_path)
            mask_np = mask_nii.get_fdata()
            mask_tensor = torch.from_numpy(np.round(mask_np).astype(np.uint8)).unsqueeze(0)

            for label, name in label_to_name.items():
                organ_mask = (mask_tensor == label).float()

                if organ_mask.sum() == 0:
                    continue  # Skip if organ not present

                print('image:', image.shape, 'mask:', organ_mask.shape)

                if image.shape != organ_mask.shape:
                    print('Different shape between image and mask')
                    image, organ_mask = pad_to_match(image, organ_mask)
                    print('image:', image.shape, 'mask:', organ_mask.shape)

                patches = crop_patch_from_mask(image, organ_mask, patch_size=(64, 64, 32), name=name)

                for patch_tensor, patch_name in patches:
                    print(patch_tensor.shape)
                    print(patch_name)
                    
                    #patch_np = patch_tensor.squeeze(0).cpu().numpy()
                    #patch_affine = np.eye(4)
                    #patch_nii = nib.Nifti1Image(patch_np, affine=patch_affine)
                    #patch_filename = os.path.join(patches_dir, f"{file}_ROI_{patch_name}.nii.gz")
                    #nib.save(patch_nii, patch_filename)

                    with torch.no_grad():
                        patch_tensor = patch_tensor.to('cuda')
                        output = model(patch_tensor.unsqueeze(0))[-1]
                        feature_vector = torch.nn.functional.adaptive_avg_pool3d(output, 1).squeeze()
                        feature_vector = feature_vector.cpu().numpy()
                        formatted_feature_vector = "[" + ", ".join(map(str, feature_vector)) + "]"

                    writer.writerow({
                        "FileName": file,
                        "ROI": patch_name,
                        "deepfeatures": formatted_feature_vector
                    })

                print(f"[{file} - {name}] GPU memory: {torch.cuda.memory_allocated() / 1e6:.2f} MB")

                torch.cuda.empty_cache()
                

    print("Organ-based feature extraction completed!")

# Paths
if __name__ == "__main__":
    masks_dir = "/mnt/nas7/data/maria/final_features/CT-ORG/masks"
    volumes_dir = "/mnt/nas7/data/maria/final_features/CT-ORG/volumes"
    output_dir = "/mnt/nas7/data/maria/final_features/ct-fm/ct-org"
    os.makedirs(output_dir, exist_ok=True)

    run_inference(volumes_dir, masks_dir, output_dir)


############# PLOTS #############

# Load CSV
csv_path = '/mnt/nas7/data/maria/final_features/ct-fm/ct-org/features_ct-fm_org_v3.csv'
df = pd.read_csv(csv_path)

# deep features from string to float lists
df['deepfeatures'] = df['deepfeatures'].apply(ast.literal_eval)

X = np.array(df['deepfeatures'].tolist())   # features array

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Group ROIs
df['ROI_grouped'] = df['ROI'].str.replace(r'_(left|right)$', '', regex=True)
grouped_labels = df['ROI_grouped']

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

plt.title("CT-FM features on CT-ORG data (UMAP)")
#plt.legend()
plt.grid(True)

output_path = os.path.join(os.path.dirname(csv_path), 'ct-fm_org_umap_final2.png')
plt.savefig(output_path)
plt.close()
