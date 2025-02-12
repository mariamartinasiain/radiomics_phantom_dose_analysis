import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import umap

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Define a dictionary to map method abbreviation to custom labels
method_labels = {
    'pyradiomics': 'Pyradiomics',
    'cnn': 'Shallow CNN',
    'swinunetr': 'SwinUNETR'
}

def apply_umap(csv_path, roi_column='ROI', series_column='SeriesDescription', feature_column='deepfeatures', output_dir='/mnt/nas7/data/maria/final_features/umap_results_dose'):
    data = pd.read_csv(csv_path)
    
    data[['Pos0', 'Pos2']] = data['SeriesDescription'].str.split('_', expand=True)[[0, 2]]  # Get scanner and dose
    
    scanners = ["A1", "A2", "B1", "B2", "G1", "G2", "C1", "H2", "D1", "E2", "F1", "E1", "H1"]

    for scanner in scanners:
        print(f'Processing scanner: {scanner}')

        # Create a directory for each scanner
        scanner_dir = os.path.join(output_dir, f'umap_scanner_{scanner}')
        os.makedirs(scanner_dir, exist_ok=True)
        
        filtered_data = data[data['Pos0'] == scanner].copy()
        filtered_data['SeriesDescription'] = filtered_data['Pos2']  # Keep just the doses
        filtered_data.drop(columns=['Pos0', 'Pos2'], inplace=True)
        num_cases = len(filtered_data)
        print(f"Number of cases: {num_cases}")

        if feature_column in filtered_data.columns and filtered_data[feature_column].dtype == 'object':
            filtered_data[feature_column] = filtered_data[feature_column].apply(lambda x: np.fromstring(x.strip("[]"), sep=','))  # Convert string to list of features
            max_len = filtered_data[feature_column].apply(len).max()
            feature_df = pd.DataFrame(filtered_data[feature_column].tolist(), index=filtered_data.index)
            feature_df.columns = [f"feature_{i}" for i in range(max_len)]
            filtered_data = pd.concat([filtered_data.drop(columns=[feature_column]), feature_df], axis=1)
        
            feature_columns = filtered_data.select_dtypes(include=[np.number]).columns.tolist()
            feature_columns = [col for col in feature_columns if col not in [roi_column, series_column]]

        def rename_duplicates(cols):
            seen = {}
            new_cols = []
            for col in cols:
                if col in seen:
                    seen[col] += 1
                    new_cols.append(f"{col}_{seen[col]}")
                else:
                    seen[col] = 0
                    new_cols.append(col)
            return new_cols

        unique_labels = sorted(filtered_data[roi_column].unique())

        colors = plt.get_cmap('viridis', len(unique_labels))
        roi_color_mapping = {label: colors(i) for i, label in enumerate(unique_labels)}
        filtered_data["ROI_numeric"] = filtered_data["ROI"].map({label: i for i, label in enumerate(unique_labels)})
        filtered_data["ROI_color"] = filtered_data["ROI_numeric"].map(roi_color_mapping)


        filtered_data.columns = [col[:28] for col in filtered_data.columns]  # to be able to save in .mat file format
        try:
            filtered_data = filtered_data.drop(columns=['Unnamed: 0'])
        except:
            print('Unammed: 0 not found!')
        filtered_data.columns = rename_duplicates(filtered_data.columns)

        if 'SpacingBetweenSlices' in filtered_data.columns:
            filtered_data = filtered_data.drop(columns=['SpacingBetweenSlices'])

        # Apply UMAP for dimensionality reduction  
        numerical_features = filtered_data.select_dtypes(include=['number'])
        features = numerical_features.values
        reducer = umap.UMAP(n_components=2, random_state=42)
        umap_results = reducer.fit_transform(features)
        
        filtered_data['UMAP_1'] = umap_results[:, 0]
        filtered_data['UMAP_2'] = umap_results[:, 1]
        
        method_key = os.path.basename(csv_path).split('_')[1].lower()
        method = method_labels.get(method_key, 'Unknown Method')
        
        # Plot UMAP results
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(umap_results[:, 0], umap_results[:, 1], c=filtered_data["ROI_numeric"], cmap='viridis')
        labels = list(roi_color_mapping.keys())  
        handles, _ = scatter.legend_elements()

        plt.legend(handles, labels, title="ROI", loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=4, frameon=False)
        plt.grid(True)
        plt.title(f"UMAP {scanner} {method}")
        plt.tight_layout()
        plt.show()

        # Save plot to output directory
        output_file = os.path.join(scanner_dir, f"{os.path.splitext(os.path.basename(csv_path))[0]}_umap_plot_{scanner}.png")
        plt.savefig(output_file, bbox_inches='tight')
        print(f"UMAP plot saved to: {output_file}")
        plt.close()


def main():
    files_dir = '/mnt/nas7/data/maria/final_features/small_roi'
    output_dir = '/mnt/nas7/data/maria/final_features/umap_results_dose'

    csv_path = [
        f'{files_dir}/features_pyradiomics_full.csv',
        f'{files_dir}/features_cnn_full.csv',
        f'{files_dir}/features_swinunetr_full.csv',
    ]

    for path in csv_path:
        apply_umap(path, output_dir=output_dir)

if __name__ == '__main__':
    main()
