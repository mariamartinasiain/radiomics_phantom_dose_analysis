import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import umap.umap_ as umap
import os
from sklearn.preprocessing import StandardScaler


def extract_mg_value(series_description):
    """Extracts dose (mg value) from the SeriesDescription column if followed by 'mGy'."""
    import re
    match = re.search(r'(\d+)mGy', series_description)
    if match:
        return int(match.group(1))
    else:
        return None

def extract_rep_number(series_description):
    """Extracts the repetition number (e.g., #1 or #9 at the end of SeriesDescription)."""
    import re
    match = re.search(r'#(\d+)$', series_description)
    if match:
        return int(match.group(1))
    else:
        return None
    
def extract_reconstruction(series_description):
    """Extracts reconstruction method (IR or FBP) from the SeriesDescription column."""
    import re
    match = re.search(r'IR|FBP', series_description)
    if match:
        return match.group(0)
    else:
        return None

def load_data(filepath):
    """Loads data and converts deep features into numerical format."""
    data = pd.read_csv(filepath)

    data['Scanner'] = data['SeriesDescription'].str.split('_', expand=True)[[0]]

    data['Dose'] = data['SeriesDescription'].apply(extract_mg_value)

    ######### Filter by dose #########
    '''
    # Filter the data for Dose = 10 mGy
    data_filtered = data[data['Dose'] == 14]
    
    feature_column='deepfeatures'

    if feature_column in data_filtered.columns and data_filtered[feature_column].dtype == 'object':
        data_filtered[feature_column] = data_filtered[feature_column].apply(lambda x: np.fromstring(x.strip("[]"), sep=','))
        max_len = data_filtered[feature_column].apply(len).max()
        feature_df = pd.DataFrame(data_filtered[feature_column].tolist(), index=data_filtered.index)
        feature_df.columns = [f"feature_{i}" for i in range(max_len)]
        data_filtered = pd.concat([data_filtered.drop(columns=[feature_column]), feature_df], axis=1)

    features = data_filtered.drop(columns=['StudyInstanceUID', 'SeriesNumber', 'SeriesDescription', 
                                'ROI', 'ManufacturerModelName', 'Manufacturer', 
                                'SliceThickness', 'SpacingBetweenSlices', 'FileName',
                                'StudyID', 'StudyDescription', 'Scanner'], errors='ignore')

    # Convert string representation of lists into actual lists
    if features.columns[0] == 'deepfeatures':
        # Clean null bytes before eval
        features['deepfeatures'] = features['deepfeatures'].astype(str).str.replace('\x00', '', regex=False)
        try:
            features = features['deepfeatures'].apply(eval).apply(pd.Series)
        except SyntaxError as e:
            print(f"Error processing {filepath}: {e}")
            raise

    return data_filtered, features
    '''

    feature_column='deepfeatures'

    if feature_column in data.columns and data[feature_column].dtype == 'object':
        data[feature_column] = data[feature_column].apply(lambda x: np.fromstring(x.strip("[]"), sep=','))
        max_len = data[feature_column].apply(len).max()
        feature_df = pd.DataFrame(data[feature_column].tolist(), index=data.index)
        feature_df.columns = [f"feature_{i}" for i in range(max_len)]
        data = pd.concat([data.drop(columns=[feature_column]), feature_df], axis=1)

    features = data.drop(columns=['StudyInstanceUID', 'SeriesNumber', 'SeriesDescription', 
                                'ROI', 'ManufacturerModelName', 'Manufacturer', 
                                'SliceThickness', 'SpacingBetweenSlices', 'FileName',
                                'StudyID', 'StudyDescription', 'Scanner', 'Dose'], errors='ignore')

    # Convert string representation of lists into actual lists
    if features.columns[0] == 'deepfeatures':
        # Clean null bytes before eval
        features['deepfeatures'] = features['deepfeatures'].astype(str).str.replace('\x00', '', regex=False)
        try:
            features = features['deepfeatures'].apply(eval).apply(pd.Series)
        except SyntaxError as e:
            print(f"Error processing {filepath}: {e}")
            raise

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
    umap_reducer = umap.UMAP(n_neighbors=20, min_dist=0.5, n_components=2, random_state=24)
    umap_results = umap_reducer.fit_transform(features_scaled)
    print(f'Number of samples being plotted: {umap_results.shape[0]}')
    return umap_results

def plot_results(features, labels, data, color_mode, output_dir, filename_suffix=""):
    """Plots the UMAP results with color coding based on ROI, Manufacturer, and Dose."""
    base_filename = f"{output_dir}/{filename_suffix}"

    # Perform UMAP
    umap_results = perform_umap(features)

    # Determine unique labels and assign colors
    unique_labels = sorted(labels.unique())
    
    if 'roi' in color_mode.lower():
        colors = plt.get_cmap('viridis', len(unique_labels))
    elif 'manufacturer' in color_mode.lower():
        colors = plt.get_cmap('tab10', len(unique_labels))
    else:
        colors = plt.get_cmap('coolwarm', len(unique_labels))

    marker_size = 10
    if 'dose' in color_mode.lower():  # If 'Dose' is the color mode, we use continuous color mapping
        plt.figure(figsize=(10, 6))
        norm = plt.Normalize(vmin=labels.min(), vmax=labels.max())  
        scatter = plt.scatter(umap_results[:, 0], umap_results[:, 1], c=labels, cmap='coolwarm', s=marker_size, alpha=0.5, norm=norm)
        cbar = plt.colorbar(scatter)  # Add colorbar
        cbar.set_label("Dose (mGy)")  
    else:
        plt.figure(figsize=(8, 6))
        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(umap_results[mask, 0], umap_results[mask, 1], 
                        color=colors(i), s=marker_size, label=label, alpha=0.5)

    # Capitalization logic for title:
    if 'roi' in color_mode.lower():
        formatted_color_mode = color_mode.upper()  # All capital letters for ROI
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=6)
    elif 'manufacturer' in color_mode.lower():
        formatted_color_mode = color_mode.capitalize()  # Capitalize the first letter for Manufacturer
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=4)
    else:
        formatted_color_mode = color_mode.capitalize()  # Keep it as is (Dose)
        plt.legend()

    if filename_suffix == 'pyradiomics':
        formatted_suffix = 'PyRadiomics'
    elif filename_suffix == 'cnn' or filename_suffix == 'ct-fm':
        formatted_suffix = filename_suffix.upper()
    else:
        formatted_suffix = 'SwinUNETR'

    title = f"UMAP Projection {formatted_suffix} - Colored by {formatted_color_mode}"
    plt.title(title)

    plt.grid(True)
    plt.savefig(f"{base_filename}_{color_mode}_umap.png")
    

def analysis(csv_paths, output_dir):
    """Main analysis function that loads data, processes it, and generates plots."""
    print("Analyzing data...")

    for csv_path in csv_paths:
        print(f"Processing {csv_path}...")
        data, features = load_data(csv_path)

        # Extract additional information for coloring
        #data['Dose'] = data['SeriesDescription'].apply(extract_mg_value)
        data['reconstruction'] = data['SeriesDescription'].apply(extract_reconstruction)

        # Normalize label names
        data['ROI'] = data['ROI'].astype(str)
        data['Manufacturer'] = data['Manufacturer'].replace({'Siemens Healthineers': 'SIEMENS',
                                                            'SIEMENS ': 'SIEMENS',
                                                             'Philips': 'PHILIPS',
                                                             'Toshiba': 'TOSHIBA',
                                                             'Ge medical systems': 'GE MEDICAL SYSTEMS'}).astype(str)

        # Extract method name from the file name
        method_name = os.path.basename(csv_path).split('_')[0]

        # Generate plots for ROI, Manufacturer, and Dose
        for color_mode in ['ROI', 'Manufacturer', 'Dose']:
            plot_results(features, data[color_mode], data, color_mode, output_dir, filename_suffix=method_name)


if __name__ == "__main__":
    # Define file paths
    #files_dir = '/mnt/nas7/data/maria/final_features/small_roi'
    files_dir = '/mnt/nas7/data/maria/final_features'
    output_dir = '/mnt/nas7/data/maria/final_features/final_features_complete/umap/six_rois'

    os.makedirs(output_dir, exist_ok=True)

    csv_paths = [
        #f'{files_dir}/final_features_complete/features_pyradiomics_4rois.csv',
        #f'{files_dir}/final_features_complete/features_cnn_4rois.csv',
        #f'{files_dir}/final_features_complete/features_swinunetr_4rois.csv',
        #f'{files_dir}/final_features_complete/features_ct-fm_4rois.csv',

        f'{files_dir}/final_features_complete/features_pyradiomics_6rois.csv',
        f'{files_dir}/final_features_complete/features_cnn_6rois.csv',
        f'{files_dir}/final_features_complete/features_swinunetr_6rois.csv',
        f'{files_dir}/final_features_complete/features_ct-fm_6rois.csv'        
    ]
    
    # Run the analysis
    analysis(csv_paths, output_dir)



