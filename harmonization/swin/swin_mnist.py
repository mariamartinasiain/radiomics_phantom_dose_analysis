import sys
sys.path.append('/home/maria/radiomics_phantom_copy/')

import csv
import os
import numpy as np
import pandas as pd

from tqdm import tqdm
from utils import load_data, get_model, get_model_oscar
from monai.data import Dataset, DataLoader,SmartCacheDataset

import torch
from medmnist import INFO
from medmnist.dataset import OrganMNIST3D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load datasets
train_dataset = OrganMNIST3D(split="train", download=True, size=64)
val_dataset = OrganMNIST3D(split="val", download=True, size=64)
test_dataset = OrganMNIST3D(split="test", download=True, size=64)

# Combine validation and test sets for feature extraction
dataset = train_dataset + val_dataset + test_dataset
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

print(f"Total samples to extract features from: {len(dataset)}")

# OrganMNIST3D class names (from medmnist docs)
organ_labels = {int(k): v for k, v in INFO['organmnist3d']['label'].items()}

print(organ_labels)  # gives you the dictionary of label names

output_dir = "/mnt/nas7/data/maria/final_features/swinunetr/"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "organmnist_swinunetr_features.csv")



# 4. Inferencia y extracción de features
def run_inference(model):
    model.eval()
    model.to(device)

    with open(output_file, "w", newline="") as csvfile:
        fieldnames = ["Index", "Label", "ROI", "DeepFeatures"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for idx, (img, target) in enumerate(tqdm(dataloader, desc="Extracting features")):
            img = img.to(device)  # shape: (1, 1, 64, 64, 64)
            label = int(target.item())
            roi_name = organ_labels[label]

            with torch.no_grad():
                val_outputs = model.swinViT(img.float())
                latentrep = val_outputs[4]  # Shape: [1, 768, 2, 2, 2]

            latentrep = latentrep.mean(dim=[2, 3, 4])  # Average → shape: [1, 768]
            feature_vector = latentrep.squeeze(0).cpu().numpy().tolist()  # Final shape: [768]

            writer.writerow({
                "Index": idx,
                "Label": label,
                "ROI": roi_name,
                "DeepFeatures": feature_vector
            })

    print("✅ Features extracted and saved.")


def main():
    checkpoint_path = "/home/reza/radiomics_phantom/checkpoints/model_swinvit.pt"
    model = get_model(model_path=checkpoint_path)
    run_inference(model)

if __name__ == "__main__":
    main()


# === STEP 2: Visualize using t-SNE and UMAP ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import StandardScaler

# Load CSV
csv_path = "/mnt/nas7/data/maria/final_features/swinunetr/organmnist_swinunetr_features.csv"
save_dir = "/mnt/nas7/data/maria/final_features/swinunetr"
os.makedirs(save_dir, exist_ok=True)

# Load features
df = pd.read_csv(csv_path)
features = df["DeepFeatures"].apply(lambda x: np.array(eval(x))).tolist()
features = np.stack(features)
labels = df["Label"].values

# Standardize features
features_scaled = StandardScaler().fit_transform(features)

organ_labels = {int(k): v for k, v in INFO['organmnist3d']['label'].items()}

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
    plt.title(f"SwinUNETR features on OrganMNIST3D ({method_name.upper()})")

    # Custom legend with organ labels and custom colors
    handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   label=organ_labels[int(i)],
                   markerfacecolor=custom_colors[int(i)], markersize=10)
        for i in unique_labels
    ]

    plt.legend(handles=handles, title="Organ", loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"swinunetr_{method_name}.png")
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
#umap_reducer = umap.UMAP(n_components=2, random_state=42)
umap_reducer = umap.UMAP(n_neighbors=20, min_dist=0.25, n_components=2, random_state=24)

umap_result = umap_reducer.fit_transform(features_scaled)
plot_and_save(umap_result, "umap")
