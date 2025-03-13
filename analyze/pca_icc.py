import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pingouin import intraclass_corr

scanners = ["A1", "A2", "B1", "B2", "G1", "G2", "C1", "H2", "D1", "E2", "F1", "E1", "H1"]

def extract_feature_type(csv_path):
    """Extracts feature type from file name."""
    return os.path.basename(csv_path).split('_')[1]


def apply_pca_to_all(data, feature_columns, output_dir, feature_type, variance_threshold=0.90):
    """Applies PCA to the entire dataset before splitting by scanner."""
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(data[feature_columns])

    pca = PCA()
    pca_features = pca.fit_transform(normalized_features)
    
    # Calculate cumulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    # Find the number of components that explain at least `variance_threshold` of the variance
    num_components = np.argmax(cumulative_variance >= variance_threshold) + 1
    print('Num components: ', num_components)
    
    # Now select the components that explain the required variance
    pca = PCA(n_components=num_components)
    pca_features = pca.fit_transform(normalized_features)

    # Create PCA feature names
    pca_columns = [f"PCA_{i+1}" for i in range(pca_features.shape[1])]
    pca_df = pd.DataFrame(pca_features, columns=pca_columns, index=data.index)

    # Plot PCA explained variance    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_components + 1), np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title(f'PCA Explained Variance (90% variance) {feature_type}')
    plt.grid()
    output_path = os.path.join(output_dir, f'explained_variance_{feature_type}.png')
    plt.savefig(output_path)
    print(f"Saving explained variance plot to: {output_path}")
    plt.close()

    # Scree Plot
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_components + 1), pca.explained_variance_ratio_, marker='o', linestyle='--')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title(f'Scree Plot {feature_type}')
    plt.grid()
    output_path = os.path.join(output_dir, f'scree_{feature_type}.png')
    plt.savefig(output_path)
    print(f"Saving scree plot to: {output_path}")
    plt.close()

    # 2D scatter plot for first two principal components
    plt.figure(figsize=(8, 8))
    # Removed cmap since it's not needed
    plt.scatter(pca_features[:, 0], pca_features[:, 1])  # No color argument here
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title(f'PCA 2D Scatter Plot {feature_type}')
    output_path = os.path.join(output_dir, f'2d_scatter_{feature_type}.png')
    plt.savefig(output_path)
    print(f"Saving 2D scatter plot to: {output_path}")
    plt.close()

    return pca, scaler, pca_df


def calculate_icc(csv_path, roi_column='ROI', series_column='SeriesDescription', feature_column='deepfeatures'):
    """Computes ICC for PCA-transformed features across scanners."""
    data = pd.read_csv(csv_path)
    feature_type = extract_feature_type(csv_path)

    # Extract scanner and dose info
    data[['Scanner', 'Dose']] = data['SeriesDescription'].str.split('_', expand=True)[[0, 2]]

    output_dir = '/mnt/nas7/data/maria/final_features/icc_results_dose/four_rois/pca'
    os.makedirs(output_dir, exist_ok=True)

    # Convert deep features if necessary
    if feature_column in data.columns and data[feature_column].dtype == 'object':
        data[feature_column] = data[feature_column].apply(lambda x: np.fromstring(x.strip("[]"), sep=','))
        max_len = data[feature_column].apply(len).max()
        feature_df = pd.DataFrame(data[feature_column].tolist(), index=data.index)
        feature_df.columns = [f"feature_{i}" for i in range(max_len)]
        data = pd.concat([data.drop(columns=[feature_column]), feature_df], axis=1)

    # Remove unwanted columns from the data
    columns_to_remove = ['Unnamed: 0', 'SpacingBetweenSlices', 'SliceThickness']
    data = data.drop(columns=[col for col in columns_to_remove if col in data.columns])

    # Now, update feature_columns after removing unwanted columns
    feature_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    feature_columns = [col for col in feature_columns if col not in [roi_column, series_column]]

    # Apply PCA globally
    pca, scaler, pca_df = apply_pca_to_all(data, feature_columns, output_dir, feature_type)

    # Replace original features with PCA-transformed ones
    data = pd.concat([data.drop(columns=feature_columns), pca_df], axis=1)

    summary = []
    icc_results_per_scanner = {}

    for scanner in scanners:
        print('Computing ICC for scanner', scanner)
        scanner_dir = os.path.join(output_dir, f'icc_scanner_{scanner}')
        os.makedirs(scanner_dir, exist_ok=True)

        print(f'Processing scanner: {scanner}')
        filtered_data = data[data['Scanner'] == scanner].copy()
        filtered_data.drop(columns=['Scanner', 'Dose'], inplace=True)

        num_cases = len(filtered_data)
        print(f"Number of cases: {num_cases}")

        # Select only numerical features
        numeric_features = filtered_data.select_dtypes(include=[np.number]).columns

        # Compute ICC
        results = []
        for feature in numeric_features:
            icc_data = filtered_data[[series_column, roi_column, feature]].dropna()
            icc_data.columns = ['raters', 'targets', 'ratings']

            try:
                #icc_result = intraclass_corr(data=icc_data, raters='raters', targets='targets', ratings='ratings')
                icc_result = intraclass_corr(data=icc_data, raters='raters', targets='targets', ratings='ratings', nan_policy='omit')
                icc = max(0, icc_result.set_index('Type').at['ICC3k', 'ICC'])
                results.append({'Feature': feature, 'ICC': icc})
            except Exception as e:
                print(f"Error for Scanner {scanner} - Feature {feature}: {e}")
                results.append({'Feature': feature, 'ICC': np.nan})

        icc_results_sorted = pd.DataFrame(results).sort_values(by='ICC', ascending=False)

        # Filter out non-numeric values
        unwanted_values = ["SpacingBetweenSlices", "SliceThickness", "ManufacturerModelName", 
                           "Manufacturer", "StudyInstanceUID", "SeriesNumber", "SeriesDescription", "ROI"]
        icc_results_sorted = icc_results_sorted[~icc_results_sorted['Feature'].isin(unwanted_values)]
        icc_results_sorted = icc_results_sorted.dropna()

        # Save the results
        base_filename = os.path.basename(csv_path).replace('.csv', '')
        try:
            icc_results_sorted.to_csv(os.path.join(scanner_dir, f'icc_dose_{base_filename}_{scanner}_pca.csv'), index=False)
        except Exception as e:
            print(f"Error saving ICC results: {e}")

        icc_results_per_scanner[(csv_path, scanner)] = icc_results_sorted
        summary.append({'Scanner': scanner, 'Num_Cases': num_cases, 'Num_Features': len(icc_results_sorted)})

    return summary, icc_results_per_scanner

def main():
    files_dir = '/mnt/nas7/data/maria/final_features/small_roi'
    output_dir = '/mnt/nas7/data/maria/final_features/icc_results_dose/four_rois/pca'
    os.makedirs(output_dir, exist_ok=True)

    csv_paths = [
        #f'{files_dir}/features_pyradiomics_full.csv',
        f'{files_dir}/features_cnn_full.csv',
        #f'{files_dir}/features_swinunetr_full.csv',
        f'{files_dir}/features_ct-fm_full.csv'
    ]

    for path in csv_paths:
        summaries, icc_results_per_scanner = calculate_icc(path)

        # Save final summary
        summary_df = pd.DataFrame(summaries)
        summary_df.to_csv(f'{output_dir}/icc_summary_{os.path.basename(path)}', index=False)

if __name__ == '__main__':
    main()
