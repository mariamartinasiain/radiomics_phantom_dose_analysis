# === STEP 1: Extract Features from OrganMNIST3D using SegResEncoder ===
import medmnist
from medmnist import INFO
from medmnist.dataset import OrganMNIST3D
from torch.utils.data import DataLoader
import torch
import csv
import os
from tqdm import tqdm
from monai.transforms import ScaleIntensityRange, EnsureType, Compose
from lighter_zoo import SegResEncoder

# 1. Load datasets
train_dataset = OrganMNIST3D(split="train", download=True, size=64)
val_dataset = OrganMNIST3D(split="val", download=True, size=64)
test_dataset = OrganMNIST3D(split="test", download=True, size=64)

# Combine validation and test sets for feature extraction
dataset = train_dataset + val_dataset + test_dataset
full_loader = DataLoader(dataset, batch_size=1, shuffle=False)

print(f"Total samples to extract features from: {len(dataset)}")

# OrganMNIST3D class names (from medmnist docs)
organ_labels = {int(k): v for k, v in INFO['organmnist3d']['label'].items()}

print(organ_labels)  # gives you the dictionary of label names

# 2. Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SegResEncoder.from_pretrained("project-lighter/ct_fm_feature_extractor").to(device)
model.eval()

# 3. Preprocessing
preprocess = Compose([
    EnsureType(),
    ScaleIntensityRange(
        a_min=-1024,
        a_max=2048,
        b_min=0,
        b_max=1,
        clip=True
    )
])

# 4. Output CSV setup
output_dir = "/mnt/nas7/data/maria/final_features/ct-fm/organmnist"
os.makedirs(output_dir, exist_ok=True)
csv_path = os.path.join(output_dir, "organmnist_ct-fm_features.csv")

# 5. Extract features and write to CSV
with open(csv_path, "w", newline="") as csvfile:
    fieldnames = ["Label", "ROI", "DeepFeatures"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for idx, (img, target) in enumerate(tqdm(full_loader, desc="Extracting features")):
        label = int(target.item())  # Use correct label from the dataloader
        organ_name = organ_labels.get(label, "Unknown Organ")  # Get organ name from dictionary
        img = preprocess(img.to(torch.float32)).to(device)

        with torch.no_grad():
            out = model(img)[-1]  # Final layer output
            pooled = torch.nn.functional.adaptive_avg_pool3d(out, 1).squeeze()
            vector = pooled.cpu().numpy()
            formatted_vector = "[" + ", ".join([f"{x:.6f}" for x in vector]) + "]"

        writer.writerow({
            "Label": label,
            "ROI": organ_name,
            "DeepFeatures": formatted_vector
        })

print(f"✅ Features saved at: {csv_path}")



# === STEP 2: Visualize using t-SNE and UMAP ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import StandardScaler

# Load CSV
csv_path = "/mnt/nas7/data/maria/final_features/ct-fm/organmnist/organmnist_ct-fm_features.csv"
save_dir = "/mnt/nas7/data/maria/final_features/ct-fm/organmnist"
os.makedirs(save_dir, exist_ok=True)

# Load features
df = pd.read_csv(csv_path)
features = df["DeepFeatures"].apply(lambda x: np.array(eval(x))).tolist()
features = np.stack(features)
labels = df["Label"].values

organ_labels = {int(k): v for k, v in INFO['organmnist3d']['label'].items()}

# Standardize features
features_scaled = StandardScaler().fit_transform(features)

custom_colors = [
    "#1f77b4",  # 0: azul (blue)
    "#aec7e8",  # 1: azul claro (light blue)
    "#ff7f0e",  # 2: naranja (orange)
    "#ffbb78",  # 3: naranja claro (light orange)
    "#2ca02c",  # 4: verde (green)
    "#98df8a",  # 5: verde claro (light green)
    "#d62728",  # 6: rojo (red)
    "#ff9896",  # 7: rosa (pink)
    "#9467bd",  # 8: morado (purple)
    "#c5b0d5",  # 9: morado claro (light purple)
    "#8c564b",  # 10: marrón (brown)
]

# Plotting function with legend
def plot_and_save(reduced, method_name):
    plt.figure(figsize=(10, 8))
    
    # Use custom colors based on label
    unique_labels = np.unique(labels)
    color_array = [custom_colors[int(l)] for l in labels]

    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=color_array, s=60, alpha=0.7)
    plt.title(f"CT-FM features on OrganMNIST3D ({method_name.upper()})")

    # Custom legend with organ labels and custom colors
    handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   label=organ_labels[int(i)],
                   markerfacecolor=custom_colors[int(i)], markersize=10)
        for i in unique_labels
    ]

    plt.legend(handles=handles, title="Organ", loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"ctfm_{method_name}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Saved {method_name.upper()} plot to: {save_path}")



# t-SNE
print("Running t-SNE...")
tsne = TSNE(n_components=2, perplexity=10, random_state=42)
tsne_result = tsne.fit_transform(features_scaled)
plot_and_save(tsne_result, "tsne")

# UMAP
print("Running UMAP...")
umap_reducer = umap.UMAP(n_neighbors=20, min_dist=0.25, n_components=2, random_state=24)

umap_result = umap_reducer.fit_transform(features_scaled)
plot_and_save(umap_result, "umap")
