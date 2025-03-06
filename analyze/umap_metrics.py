import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from tqdm import tqdm
from matplotlib.lines import Line2D
import umap.umap_ as umap
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Define markers for visualization
markers = ['o', 's', 'D', '^', 'v', '>', '<', 'p', '*', '+', 'x']
scanners = {'A1': 'SOMATOM Definition Edge', 'A2': 'SOMATOM Definition Flash', 
            'B1': 'SOMATOM X.cite', 'B2': 'SOMATOM Edge Plus', 'C1': 'iCT 256', 
            'D1': 'Revolution EVO', 'E1': 'Aquilion Prime SP', 'E2': 'GE MEDICAL SYSTEMS', 
            'F1': 'BrightSpeed S', 'G1': 'SOMATOM Definition Edge', 'G2': 'SOMATOM Definition Flash', 
            'H1': 'Aquilion', 'H2': 'Brilliance 64'}

def extract_mg_value(series_description):
    """Extracts dose (mg value) from the SeriesDescription column if followed by 'mGy'."""
    import re
    match = re.search(r'(\d+)mGy', series_description)
    return int(match.group(1)) if match else None

def extract_reconstruction(series_description):
    """Extracts reconstruction method (IR or FBP) from the SeriesDescription column."""
    import re
    match = re.search(r'IR|FBP', series_description)
    return match.group(0) if match else None

def load_data(filepath):
    """Loads data and converts deep features into numerical format."""
    data = pd.read_csv(filepath)
    try:
        data['deepfeatures'] = data['deepfeatures'].apply(lambda x: np.fromstring(x.strip("[]"), sep=','))
    except:
        print('Deep features not found, trying with pyradiomics.')

    features = data.drop(columns=['StudyInstanceUID', 'SeriesNumber', 'SeriesDescription', 
                                  'ROI', 'ManufacturerModelName', 'Manufacturer', 
                                  'SliceThickness', 'SpacingBetweenSlices'], errors='ignore')

    if features.columns[0] == 'deepfeatures':
        features = features['deepfeatures'].apply(eval).apply(pd.Series)

    return data, features

def features_to_numpy(features):
    """Converts features into a NumPy array for UMAP."""
    try:
        features_array = np.zeros([len(features), len(features['deepfeatures'].iloc[0])])
    except:
        return np.array(features)
    for i, row in enumerate(features['deepfeatures']):
        features_array[i] = row
    return features_array

def perform_umap(features):
    """Performs UMAP dimensionality reduction."""
    features_array = features_to_numpy(features)
    features_scaled = StandardScaler().fit_transform(features_array)
    umap_reducer = umap.UMAP(n_neighbors=25, min_dist=0.5, n_components=2, random_state=42)
    umap_results = umap_reducer.fit_transform(features_scaled)
    return umap_results

def compute_metrics(umap_results, labels, method_name, color_mode, all_metrics):
    """Computes clustering metrics and stores them in a list."""
    metrics = {
        'Method': method_name,
        'Color Mode': color_mode,
        'Silhouette Score': silhouette_score(umap_results, labels) if len(set(labels)) > 1 else np.nan,
        'Davies-Bouldin Score': davies_bouldin_score(umap_results, labels),
        'Calinski-Harabasz Score': calinski_harabasz_score(umap_results, labels)
    }
    all_metrics.append(metrics)

def plot_results(features, labels, color_mode, output_dir, method_name, all_metrics):
    """Plots the UMAP results with color coding and computes metrics."""
    base_filename = f"{output_dir}/{method_name}"

    # Perform UMAP
    umap_results = perform_umap(features)

    # Compute metrics
    compute_metrics(umap_results, labels, method_name, color_mode, all_metrics)

    # Determine unique labels and assign colors
    unique_labels = sorted(labels.unique())
    
    if 'roi' in color_mode.lower():
        colors = plt.get_cmap('viridis', len(unique_labels))
    elif 'manufacturer' in color_mode.lower():
        colors = plt.get_cmap('tab10', len(unique_labels))
    else:
        colors = plt.get_cmap('Spectral', len(unique_labels))

    plt.figure(figsize=(8, 6))

    if 'dose' in color_mode.lower():  
        norm = plt.Normalize(vmin=labels.min(), vmax=labels.max())  
        scatter = plt.scatter(umap_results[:, 0], umap_results[:, 1], c=labels, cmap='Spectral', alpha=0.5, norm=norm)
        cbar = plt.colorbar(scatter)
        cbar.set_label("Dose (mGy)")  
    else:
        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(umap_results[mask, 0], umap_results[mask, 1], 
                        color=colors(i), label=label, alpha=0.5)

    formatted_color_mode = color_mode.upper() if 'roi' in color_mode.lower() else color_mode.capitalize()
    formatted_method = method_name.capitalize() if method_name == 'pyradiomics' else method_name.upper()

    title = f"UMAP Projection {formatted_method} - Colored by {formatted_color_mode}"
    plt.title(title)
    plt.legend()
    #plt.savefig(f"{base_filename}_{color_mode}_umap.png")
    plt.show()

def analysis(csv_paths, output_dir):
    """Main analysis function that processes data and generates plots."""
    print("Analyzing data...")
    all_metrics = []  # Store metrics for all methods

    for csv_path in csv_paths:
        print(f"Processing {csv_path}...")
        data, features = load_data(csv_path)

        # Extract additional information
        data['Dose'] = data['SeriesDescription'].apply(extract_mg_value)
        data['Reconstruction'] = data['SeriesDescription'].apply(extract_reconstruction)

        # Normalize label names
        data['ROI'] = data['ROI'].astype(str).str.capitalize()
        data['Manufacturer'] = data['Manufacturer'].replace({'Siemens Healthineers': 'SIEMENS', 
                                                             'Philips': 'PHILIPS'}).astype(str).str.capitalize()

        # Extract method name from filename
        method_name = os.path.basename(csv_path).split('_')[1]

        # Generate plots for ROI, Manufacturer, and Dose
        for color_mode in ['ROI', 'Manufacturer', 'Dose']:
            plot_results(features, data[color_mode], color_mode, output_dir, method_name, all_metrics)

    # Save all metrics into one file
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(f"{output_dir}/all_metrics.csv", index=False)
    print(f"Saved all metrics to {output_dir}/all_metrics.csv")

if __name__ == "__main__":
    files_dir = '/mnt/nas7/data/maria/final_features/small_roi'
    output_dir = '/mnt/nas7/data/maria/final_features/umap_results_dose'
    os.makedirs(output_dir, exist_ok=True)

    csv_paths = [
        f'{files_dir}/features_pyradiomics_full.csv',
        f'{files_dir}/features_cnn_full.csv',
        f'{files_dir}/features_swinunetr_full.csv',
    ]
    
    analysis(csv_paths, output_dir)


