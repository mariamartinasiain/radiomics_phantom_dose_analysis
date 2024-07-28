import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from sklearn.metrics import silhouette_score
import os

markers = ['o', 's', 'D', '^', 'v', '>', '<', 'p', '*', '+', 'x']
scanners = {'A1' : 'SOMATOM Definition Edge', 'A2':'SOMATOM Definition Flash', 'B1':'SOMATOM X.cite', 'B2':'SOMATOM Edge Plus', 'C1':'iCT 256', 'D1':'Revolution EVO', 'E1':'Aquilion Prime SP', 'E2':'GE MEDICAL SYSTEMS', 'F1':'BrightSpeed S', 'G1':'SOMATOM Definition Edge', 'G2':'SOMATOM Definition Flash', 'H1':'Aquilion', 'H2':'Brilliance 64'}
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

def extract_rep_number(series_description):
    import re
    # Recherche d'un motif numérique du style #1 ou #9 finissant series_description
    match = re.search(r'#(\d+)$', series_description)
    if match:
        return int(match.group(1))
    else:
        return None
    
def extract_rep_number2(series_description):
    import re
    # Recherche d'un motif numérique du style #1 ou #9 finissant series_description
    match = re.search(r'.*(IR|FBP).*#(\d+)$', series_description)
    if match:
        #print(match.group(1))
        if match.group(1) == 'IR':
            return int(match.group(2)) + 100
        else:
            return int(match.group(2))
    else:
        return None
    
def extract_recontruction(series_description):
    import re
    # Recherche d'un motif du style IR ou FBP
    match = re.search(r'IR|FBP', series_description)
    if match:
        return match.group(0)
    else:
        return None

def generate_advanced_markers(num_required):
    base_markers = ['o', 'X', 'H','s', '*','D', '^', 'v', '>', '<', 'p' ]
    line_styles = ['-', '--', '-.', ':']
    marker_sizes = [3]  
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

def load_data(filepath, color_mode='roi', mg_filter=None,rep_filter=None):
    data = pd.read_csv(filepath)
    
    # Filtrage sur la quantité de mg si spécifié
    
    
    
    
    features = data.drop(columns=['StudyInstanceUID', 'SeriesNumber', 'SeriesDescription', 'ROI','ManufacturerModelName','Manufacturer','SliceThickness','SpacingBetweenSlices'],errors='ignore')
    #verifier si features est plutot une liste ou un string d'une liste
    if features.columns[0] == 'deepfeatures':
        features = features['deepfeatures'].apply(eval).apply(pd.Series)
    if color_mode == 'roi':
        labels = data['ROI']
        supp_info = data['SeriesNumber']
    elif color_mode == 'series_desc':
        # Extraction des deux premiers caractères de la SeriesDescription
        labels = data['SeriesDescription'].str[:2].map(scanners)
        supp_info = data['SeriesNumber']
    if color_mode == 'all':
        labels = data['ROI']
        supp_info = data['SeriesDescription'].str[:2]
    if color_mode == 'manufacturer':
        labels = data['Manufacturer']
        supp_info = data['SeriesNumber']
    if color_mode == 'mg':
        data['mg_value'] = data['SeriesDescription'].apply(extract_mg_value)
        labels = data['mg_value']
        supp_info = data['SeriesNumber']
    if color_mode == 'reconstruction':
        data['reconstruction'] = data['SeriesDescription'].apply(extract_recontruction)
        labels = data['reconstruction']
        supp_info = data['SeriesNumber']
    #if color_mode == 'reconstruction':
    
    print(f"Loaded {len(features)} features with a size of {len(features.columns)}")
    print(f"Loaded {len(labels)} labels")
    data['mg_value'] = data['SeriesDescription'].apply(extract_mg_value)
    if mg_filter is not None:
        data = data[data['mg_value'] == mg_filter]
        
    data['SeriesNumber'] = data['SeriesDescription'].apply(extract_rep_number)
    if rep_filter is not None:
        data = data[data['SeriesNumber'] == rep_filter]
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

def save_silhouette_score(scores_filename, datasetname, color_mode, mg_filter, silhouette_avg):
    if not os.path.exists(scores_filename):
        with open(scores_filename, 'w') as f:
            f.write("dataset,color_mode,mg_filter,silhouette_score\n")
    with open(scores_filename, 'a') as f:
        if isinstance(silhouette_avg, str):
            f.write(f"{datasetname},{color_mode},{mg_filter},{silhouette_avg}\n")
        else:
            f.write(f"{datasetname},{color_mode},{mg_filter},{silhouette_avg:.4f}\n")

def analysis(color_mode='series_desc', mg_filter=None, filepath='../../all_dataset_features/averaged_swin_deepfeatures.csv',datasetname='averaged_swin_deepfeatures',rep_filter=None):
    print("Analyzing data...")

    features, labels, supp_info = load_data(filepath, color_mode, mg_filter=mg_filter,rep_filter=rep_filter)
    #silhouette_avg = silhouette_score(features, labels)
    #print(f'Silhouette Score on the whole feature space: {silhouette_avg:.4f}')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    scores_filename = f"{filepath.rsplit('/', 1)[0]}/silhouette_scores_{timestamp}.csv"
    #save_silhouette_score(scores_filename, timestamp, datasetname, color_mode, mg_filter, silhouette_avg)
    
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
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(f"{datasetname}_{color_mode}_{mg_filter}_{rep_filter}_PCA.png")
    #plt.show()

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
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(f"{datasetname}_{color_mode}_{mg_filter}_{rep_filter}_tSNE.png")
    #plt.show()
    
#only doing the silhouette score analysis
def silhouette_score_analysis(color_mode='series_desc', mg_filter=None, filepath='../../all_dataset_features/averaged_swin_deepfeatures.csv', datasetname='averaged_swin_deepfeatures'):
    print("Analyzing data...")

    features, labels, supp_info = load_data(filepath, color_mode, mg_filter=mg_filter)
    
    # Check number of unique labels
    unique_labels = labels.unique()
    if len(unique_labels) < 2:
        print("Not enough labels to calculate silhouette score.")
        silhouette_avg = -10000
    else:
        silhouette_avg = silhouette_score(features, labels)
        print(f'Silhouette Score on the whole feature space: {silhouette_avg:.4f}')
    
    # Save silhouette score to a CSV file
    scores_filename = "silhouette_scores.csv"
    save_silhouette_score(scores_filename, datasetname, color_mode, mg_filter, silhouette_avg)



    

def batch_analysis():
    color_modes = ['roi','reconstruction', 'series_desc', 'manufacturer','all']
    mg_filters = [None, 10]
    #features_files = ['../../all_dataset_features/pyradiomics_features.csv','../../all_dataset_features/swin_deepfeatures.csv','../../all_dataset_features/averaged_swin_deepfeatures.csv','../../all_dataset_features/pca_swin_deepfeatures.csv','../../all_dataset_features/new_model_swin_deepfeatures.csv','../../all_dataset_features/features_ocar_full.csv','../../all_dataset_features/features_swinunetr_full.csv','../../all_dataset_features/pyradiomics_features_full.csv']
    #datasetnames = ['pyradiomics_features','swin_deepfeatures','averaged_swin_deepfeatures','pca_swin_deepfeatures','new_model_swin_deepfeatures','features_ocar_full','features_swinunetr_full','pyradiomics_features_full']
    features_files = ['torch_normalized_deepfeaturesoscar.csv']
    datasetnames = ['torch_normalized_deepfeaturesoscar']
    for features_file in features_files:
        for color_mode in color_modes:
            for mg_filter in mg_filters:
                print(f'Analyzing {features_file} with color mode {color_mode} and mg filter {mg_filter}')
                datasetname = datasetnames[features_files.index(features_file)]
                analysis(color_mode, mg_filter, features_file, datasetname,rep_filter=None)

if __name__ == "__main__":
    batch_analysis()
