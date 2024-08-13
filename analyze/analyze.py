import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from tqdm import tqdm
from sklearn.metrics import silhouette_score
from matplotlib.lines import Line2D
from sklearn.preprocessing import LabelEncoder
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
    print(series_description)
    if match:
        print(f"Match: {match}")
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

def miniload_data(filepath,fsize=None):
    data = pd.read_csv(filepath)
    features = data.drop(columns=['StudyInstanceUID', 'SeriesNumber', 'SeriesDescription', 'ROI','ManufacturerModelName','Manufacturer','SliceThickness','SpacingBetweenSlices'],errors='ignore')
    
    #verifier si features est plutot une liste ou un string d'une liste
    if features.columns[0] == 'deepfeatures':
        features = features['deepfeatures'].apply(eval).apply(pd.Series)
    
    print(f"features shape: {features.shape}")
    
    if fsize is None:
        features = features
    elif fsize > 0:
        features = features.iloc[:, :fsize]
    else:
        features = features.iloc[:, fsize + features.shape[1]:]    
    return data, features

def load_data(filepath, color_mode='roi', mg_filter=None,rep_filter=None,self_load=True,data=None,features=None):
    
    if self_load:
        data,features = miniload_data(filepath)
    
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
        print(f"feature length before filter: {len(features)}")
        features = features[data['mg_value'] == mg_filter]
        data = data[data['mg_value'] == mg_filter]
        print(f"feature length after filter: {len(features)}")
        
    data['SeriesNumber'] = data['SeriesDescription'].apply(extract_rep_number)
    if rep_filter is not None:
        features = features[data['SeriesNumber'] == rep_filter]
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
from umap import UMAP
def perform_umap(features):
    print("Performing UMAP")
    umap = UMAP(n_components=2)
    umap_results = umap.fit_transform(features)
    return umap_results

def analysis(color_mode='series_desc', mg_filter=None, filepath='../../all_dataset_features/averaged_swin_deepfeatures.csv',datasetname='averaged_swin_deepfeatures',rep_filter=None,data=None,features=None):
    print("Analyzing data...")

    if data is not None and features is not None:
        features, labels, supp_info = load_data(filepath, color_mode, mg_filter=mg_filter,rep_filter=rep_filter,self_load=False,data=data,features=features)
    else:
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
    #plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(f"{datasetname}_{color_mode}_{mg_filter}_{rep_filter}_PCA.png")
    #plt.show()

    print("Performing umap")

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
    #plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
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

def plot_tsne(ax, X, y, title, colors, markersize=2):
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=colors, s=markersize)
    #augmenting title font size
    ax.set_title(title, fontsize=20)
    ax.grid(True, linestyle='--', linewidth=0.5)
    
    
    return scatter

# Function to plot combined t-SNE results
def plot_combined_tsne(features_list, labels_list):
    print("Performing t-SNE for all feature sets")
    
    le = LabelEncoder()
    
    tsne_results_list = [perform_tsne(features) for features in features_list]
    #tsne_results_list = [np.random.rand(100,2) for i in range(4)]
    
    encoder_labels = [le.fit_transform(labels) for labels in labels_list]
    #encoder_labels = [np.random.randint(0,6,100) for i in range(4)]
    
    unique_labels = le.classes_
    #unique_labels = ['cyst1', 'metastasis', 'hemangioma', 'normal1', 'cyst2', 'normal2']
    
    unique_labels = np.sort(unique_labels)
    #print(f"Unique labels: {unique_labels}")

    print("Plotting combined umap results")

    fig, axs = plt.subplots(2, 2, figsize=(14, 14))
    
    # Define a colormap using the updated method
    #colors = plt.cm.get_cmap('viridis', len(unique_labels))
    colors = plt.cm.get_cmap('viridis', len(unique_labels))

    scatter1 = plot_tsne(axs[0, 0], tsne_results_list[0], encoder_labels[0], 'Radiomics', colors, markersize=3)
    scatter2 = plot_tsne(axs[0, 1], tsne_results_list[1], encoder_labels[1], 'SwinUNETR Features', colors, markersize=3)
    scatter3 = plot_tsne(axs[1, 0], tsne_results_list[2], encoder_labels[2], 'Shallow CNN Features', colors, markersize=3)
    scatter4 = plot_tsne(axs[1, 1], tsne_results_list[3], encoder_labels[3], 'Contrastive SwinUNETR Features', colors, markersize=3)
    
    # Create a legend below the plots
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=colors(i), markersize=5, label=unique_labels[i]) for i in range(len(unique_labels))]
    fig.legend(handles=handles, loc='upper center', title="Labels", ncol=3, fontsize='x-large')

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.2, wspace=0.3, hspace=0.3)
    plt.show()

    

def batch_analysis():
    color_modes = ['roi','manufacturer']
    mg_filters = [10]
    fsizes = [None]#, -68,700]
    #features_files = ['../../all_dataset_features/pyradiomics_features_full.csv','../../all_dataset_features/features_swinunetr_full.csv','../../all_dataset_features/features_ocar_full.csv']
    #datasetnames = ['pyradiomics_features','swin_deepfeatures','features_oscar']
    #features_files = ['../../all_dataset_features/averaged_FT_contrastive_classification_swin_features.csv','../../all_dataset_features/averaged_FT_contrastive_reconstruction_swin_features.csv','../../all_dataset_features/averaged_FT_contrastive_swin_features.csv',
    features_files = ['../../all_dataset_features/features_swinunetr_full.csv']
    #datasetnames = ['averaged_FT_contrastive_classification_swin_features','averaged_FT_contrastive_reconstruction_swin_features','averaged_FT_contrastive_swin_features','
    datasetnames = ['features_swinunetr_full']
    for features_file in features_files:
        for fs in fsizes:
            print(f'Loading data from {features_file}')
            data, features = miniload_data(features_file,fsize=fs)
            print(f'Loaded {len(features)} features with a size of {len(features.columns)}')
            for mg_filter in mg_filters:
                for color_mode in color_modes:
                    print(f'Analyzing {features_file} with color mode {color_mode} and mg filter {mg_filter}')
                    datasetname = datasetnames[features_files.index(features_file)]
                    datasetname = f"{datasetname}@{fs}@"
                    analysis(color_mode, mg_filter, features_file, datasetname,rep_filter=None,data=data,features=features)

def plots_paper():
    color_mode = 'manufacturer'
    mg_filter = 10
    fs = None
    
    features_files = ['../../all_dataset_features/pyradiomics_features_full.csv',
                      '../../all_dataset_features/features_swinunetr_full.csv',
                      '../../all_dataset_features/features_ocar_full.csv',
                      '../../all_dataset_features/paper_contrastive_F1_features2.csv']
                      
    datasetnames = ['pyradiomics_features', 'swin_deepfeatures', 'features_oscar', 'paper_contrastive__features']
    
    features_list = []
    labels_list = []
    supp_info_list = []
    
    for features_file in features_files:
        print(f'Loading data from {features_file}')
        data, features = miniload_data(features_file, fsize=fs)
        print(f'Loaded {len(features)} features with a size of {len(features.columns)}')
        
        print(f'Analyzing {features_file} with color mode {color_mode} and mg filter {mg_filter}')
        datasetname = datasetnames[features_files.index(features_file)]
        
        features, labels, supp_info = load_data(features_file, color_mode, mg_filter=mg_filter, rep_filter=None, self_load=False, data=data, features=features)
        
        print(f"Labels head: {labels.head()}")
        
        labels = labels.copy()
        labels = labels.replace('metastatsis', 'metastasis')
        labels = labels.replace('Siemens Healthineers', 'SIEMENS')
        
        features_list.append(features)
        labels_list.append(labels)
        supp_info_list.append(supp_info)
    
    # Example titles for the four plots
    titles = ['Radiomics', 'SwinUNETR Features', 'Shallow CNN Features', 'Contrastive SwinUNETR Features']

    # Output path for the final plot
    output_path_final_adjusted_margin = "/mnt/data/combined_tsne_plots_final_adjusted_margin_tight.png"

    # Call the function with your data
    plot_combined_tsne(features_list, labels_list)


if __name__ == "__main__":
    batch_analysis()
