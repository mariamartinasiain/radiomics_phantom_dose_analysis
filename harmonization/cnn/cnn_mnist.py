import sys
sys.path.append('/home/maria/radiomics_phantom_copy/')
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import os
import numpy as np
import csv
from tqdm import tqdm
from monai.transforms import Compose, LoadImage, EnsureType
from monai.data import SmartCacheDataset, DataLoader,ThreadDataLoader
import torch
import tensorflow as tf2
from medmnist import INFO
from medmnist.dataset import OrganMNIST3D

gpus = tf2.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf2.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# 1. Load datasets
train_dataset = OrganMNIST3D(split="train", download=True, size=64)
val_dataset = OrganMNIST3D(split="val", download=True, size=64)
test_dataset = OrganMNIST3D(split="test", download=True, size=64)

# Combine validation and test sets for feature extraction
dataset = train_dataset + val_dataset + test_dataset
full_loader = DataLoader(dataset, batch_size=1, shuffle=False)

print(f"📦 Total samples to extract features from: {len(dataset)}")

# OrganMNIST3D class names (from medmnist docs)
organ_labels = {int(k): v for k, v in INFO['organmnist3d']['label'].items()}

print(organ_labels)  # gives you the dictionary of label names


def run_inference():

    model_dir = '/mnt/nas7/data/maria/final_features/QA4IQI/QA4IQI/'
    model_file = model_dir + 'organs-5c-30fs-acc92-121.meta'

    # Start a new session
    sess = tf.Session()

    # Load the graph
    saver = tf.train.import_meta_graph(model_file)
    saver.restore(sess, tf.train.latest_checkpoint(model_dir))

    # Access the graph
    graph = tf.get_default_graph()
    feature_tensor = graph.get_tensor_by_name('MaxPool3D_1:0')

    #names for input data and dropout:
    x = graph.get_tensor_by_name('x_start:0') 
    keepProb = graph.get_tensor_by_name('keepProb:0')  
    
    
    try:
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

        # Calculer le nombre total de poids
        total_params = sum([sess.run(tf.size(var)) for var in variables])
        print(f"Le nombre total de poids dans le modèle est : {total_params}")
    except:
        print("Error while calculating the number of parameters in the model")
    device_id = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    torch.cuda.set_device(device_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    transforms = Compose([
    EnsureType(),                         # Ensure correct data type
    ])


    output_dir = "/mnt/nas7/data/maria/final_features/cnn/"
    output_file = os.path.join(output_dir, "organmnist_cnn_features.csv")


    # Open CSV file once
    with open(output_file, "w", newline="") as csvfile:
        fieldnames = ["Label", "ROI", "DeepFeatures"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for idx, (img, target) in enumerate(tqdm(full_loader, desc="🔍 Extracting features")):
            label = int(target.item())  # Use correct label from the dataloader
            organ_name = organ_labels.get(label, "Unknown Organ")  # Get organ name from dictionary
            img = img[:, :, :32]  # Crop the depth to 32 slices
            img = transforms(img.to(torch.float32)).to(device)
            #patch = patch.to(device)    # Move tensor to the GPU
            flattened_image = img.flatten()
            flattened_image = flattened_image.cpu().numpy()  # Ensure it's a NumPy array
            flattened_image = flattened_image.reshape(1, -1)  # Reshape to (1, 131072)

            features = sess.run(feature_tensor, feed_dict={x: flattened_image, keepProb: 1.0})

            latentrep = sess.run(tf.reshape(features, [-1])).tolist()

            # Write to CSV
            writer.writerow({
                "Label": label,
                "ROI": organ_name,
                "DeepFeatures": latentrep
            })

                 
    print("Done!")


if __name__ == "__main__":
    run_inference()


# === STEP 2: Visualize using t-SNE and UMAP ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import StandardScaler

# Load CSV
csv_path = "/mnt/nas7/data/maria/final_features/cnn/organmnist_cnn_features.csv"
save_dir = "/mnt/nas7/data/maria/final_features/cnn"
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
    plt.title(f"CNN features on OrganMNIST3D ({method_name.upper()})")

    # Custom legend with organ labels and custom colors
    handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   label=organ_labels[int(i)],
                   markerfacecolor=custom_colors[int(i)], markersize=10)
        for i in unique_labels
    ]

    plt.legend(handles=handles, title="Organ", loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"cnn_{method_name}.png")
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
