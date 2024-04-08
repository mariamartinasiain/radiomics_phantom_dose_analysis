import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

markers = ['o', 's', 'D', '^', 'v', '>', '<', 'p', '*', '+', 'x']

def extract_mg_value(series_description):
    """
    Extracts the milligram (mg) value from the SeriesDescription column.
    Assumes that the mg value is followed by 'mGy'.
    """
    import re
    # Recherche d'un motif numérique suivi de 'mGy'
    match = re.search(r'(\d+)mGy', series_description)
    if match:
        return int(match.group(1))
    else:
        return None

def generate_advanced_markers(num_required):
    base_markers = ['o', 'X', 'H','s', '*','D', '^', 'v', '>', '<', 'p' ]
    line_styles = ['-', '--', '-.', ':']
    marker_sizes = [6]  
    filled_markers = [True, False]

    generated_markers = []

    for filled in filled_markers:
        for size in marker_sizes:
            for line_style in line_styles:
                    for marker in base_markers:
                        if len(generated_markers) < num_required:
                            generated_markers.append({
                                'marker': marker,
                                'linestyle': line_style,
                                'size': size,
                                'filled': filled
                            })
                        else:
                            return generated_markers
    while len(generated_markers) < num_required:
        #duplicating markers
        generated_markers = generated_markers + generated_markers
        

    return generated_markers

def load_data(filepath, color_mode='roi', mg_filter=None):
    data = pd.read_csv(filepath)
    
    # Filtrage sur la quantité de mg si spécifié
    data['mg_value'] = data['SeriesDescription'].apply(extract_mg_value)
    if mg_filter is not None:
        data = data[data['mg_value'] == mg_filter]
    
    features = data.drop(columns=['StudyInstanceUID', 'SeriesNumber', 'SeriesDescription', 'ROI','ManufacturerModelName','Manufacturer','SliceThickness','SpacingBetweenSlices'],errors='ignore')
    #verifier si features est plutot une liste ou un string d'une liste
    if features.columns[0] == 'deepfeatures':
        features = features['deepfeatures'].apply(eval).apply(pd.Series)
    if color_mode == 'roi':
        labels = data['ROI']
        supp_info = data['SeriesNumber']
    elif color_mode == 'series_desc':
        # Extraction des deux premiers caractères de la SeriesDescription
        labels = data['SeriesDescription'].str[:2]
        supp_info = data['SeriesNumber']
    if color_mode == 'all':
        labels = data['ROI']
        supp_info = data['SeriesDescription'].str[:2]
    
    print(f"Loaded {len(features)} features")
    print(f"Loaded {len(labels)} labels")
    #print(f"Features: {features}")
    return features, labels, supp_info

def perform_pca(features):
    features_scaled = StandardScaler().fit_transform(features)
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(features_scaled)
    explained_variance = pca.explained_variance_ratio_ * 100  # Convertir en pourcentage
    return principal_components, explained_variance

def perform_tsne(features):
    features_scaled = StandardScaler().fit_transform(features)
    perplexity = min(40, len(features_scaled) - 1)
    tsne_results = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=300).fit_transform(features_scaled)
    return tsne_results


def main(color_mode='all'):
    print("Analyzing data...")

    filepath = "../df_test.csv"
    features, labels, supp_info = load_data(filepath, color_mode)

    
    unique_labels = labels.unique()
    
    colors = plt.cm.get_cmap('viridis', len(unique_labels))
    unique_supp_info = np.unique(supp_info)
    markers = generate_advanced_markers(len(unique_supp_info)+1)
        
    marker_map = dict(zip(unique_supp_info, markers))

    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_filename = f"{filepath.rsplit('/', 1)[0]}/results_{timestamp}"

    print("Performing PCA")

    pca_results, explained_variance = perform_pca(features)
    
    plt.figure(figsize=(8, 6))
    for i, label in enumerate(unique_labels):
        for j, info in enumerate(unique_supp_info):
            mask = (labels == label) & (supp_info == info)
            marker_props = marker_map[info]
            plt.scatter(pca_results[mask, 0], pca_results[mask, 1], 
                        color=colors(i),
                        marker=marker_props['marker'], s=marker_props['size'],
                        linewidths=2,  # Adjust line width for visibility
                        linestyle=marker_props['linestyle'],
                        facecolors='none' if not marker_props['filled'] else colors(i))
            
    for i, label in enumerate(unique_labels):
        plt.scatter([], [], color=colors(i), label=label)
    
    plt.title('PCA Results')
    plt.xlabel(f'Principal Component 1 ({explained_variance[0]:.2f}%)')
    plt.ylabel(f'Principal Component 2 ({explained_variance[1]:.2f}%)')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{base_filename}_PCA.png")
    plt.show()

    print("Performing t-SNE")

    tsne_results = perform_tsne(features)


    plt.figure(figsize=(8, 6))
    for i, label in enumerate(unique_labels):
        for j, info in enumerate(unique_supp_info):
            mask = (labels == label) & (supp_info == info)
            marker_props = marker_map[info]
            plt.scatter(tsne_results[mask, 0], tsne_results[mask, 1], 
                        color=colors(i), 
                        marker=marker_props['marker'], s=marker_props['size'],
                        linewidths=2,  # Adjust line width for visibility
                        linestyle=marker_props['linestyle'],
                        facecolors='none' if not marker_props['filled'] else colors(i))
            
    for i, label in enumerate(unique_labels):
        plt.scatter([], [], color=colors(i), label=label)
    plt.title('t-SNE Results')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{base_filename}_TSNE.png")
    plt.show()
    
    


if __name__ == "__main__":
    main()
