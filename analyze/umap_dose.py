import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import umap
from sklearn.preprocessing import StandardScaler 

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Define a dictionary to map method abbreviation to custom labels
method_labels = {
    'pyradiomics': 'Pyradiomics',
    'cnn': 'Shallow CNN',
    'swinunetr': 'SwinUNETR'
}

def apply_umap(csv_path, roi_column='ROI', series_column='SeriesDescription', feature_column='deepfeatures', output_dir='/mnt/nas7/data/maria/final_features/umap_results_dose'):
    data = pd.read_csv(csv_path)
    
    # Extract scanner (Pos0) and dose (Pos2) from SeriesDescription
    data[['Pos0', 'Pos2']] = data['SeriesDescription'].str.split('_', expand=True)[[0, 2]]  

    scanners = ["A1", "A2", "B1", "B2", "G1", "G2", "C1", "H2", "D1", "E2", "F1", "E1", "H1"]

    for scanner in scanners:
        print(f'Processing scanner: {scanner}')

        # Create a directory for each scanner
        scanner_dir = os.path.join(output_dir, f'umap_scanner_{scanner}')
        os.makedirs(scanner_dir, exist_ok=True)
        
        # Filter data by scanner
        filtered_data = data[data['Pos0'] == scanner].copy()
        filtered_data['SeriesDescription'] = filtered_data['Pos2']  # Keep just the doses
        filtered_data.drop(columns=['Pos0', 'Pos2'], inplace=True)
        num_cases = len(filtered_data)
        print(f"Number of cases: {num_cases}")

        # Convert feature column from string to list of floats
        if feature_column in filtered_data.columns and filtered_data[feature_column].dtype == 'object':
            filtered_data[feature_column] = filtered_data[feature_column].apply(lambda x: np.fromstring(x.strip("[]"), sep=','))
            max_len = filtered_data[feature_column].apply(len).max()
            feature_df = pd.DataFrame(filtered_data[feature_column].tolist(), index=filtered_data.index)
            feature_df.columns = [f"feature_{i}" for i in range(max_len)]
            filtered_data = pd.concat([filtered_data.drop(columns=[feature_column]), feature_df], axis=1)

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

        # ROI-based color mapping
        unique_labels = sorted(filtered_data[roi_column].unique())
        colors = plt.get_cmap('viridis', len(unique_labels))
        roi_color_mapping = {label: colors(i) for i, label in enumerate(unique_labels)}
        filtered_data["ROI_numeric"] = filtered_data["ROI"].map({label: i for i, label in enumerate(unique_labels)})
        filtered_data["ROI_color"] = filtered_data["ROI_numeric"].map(roi_color_mapping)

        # Extract numeric part of dose and sort labels correctly
        filtered_data['dose_numeric'] = filtered_data['SeriesDescription'].str.extract('(\d+)', expand=False).astype(float)

        # Sort doses while keeping original labels (Pos2)
        sorted_doses = filtered_data[['SeriesDescription', 'dose_numeric']].drop_duplicates().sort_values(by='dose_numeric')
        unique_doses = sorted_doses['dose_numeric'].tolist()
        unique_labels = sorted_doses['SeriesDescription'].tolist()

        # Assign colors based on sorted numeric doses
        dose_colors = plt.get_cmap('coolwarm', len(unique_doses))
        dose_color_mapping = {dose: dose_colors(i) for i, dose in enumerate(unique_doses)}

        # Apply colors to data
        filtered_data["dose_color"] = filtered_data["dose_numeric"].map(dose_color_mapping)

        # Trim long column names for saving compatibility
        filtered_data.columns = [col[:28] for col in filtered_data.columns]
        try:
            filtered_data = filtered_data.drop(columns=['Unnamed: 0'])
        except:
            print('Unammed: 0 not found!')
        filtered_data.columns = rename_duplicates(filtered_data.columns)

        if 'SpacingBetweenSlices' in filtered_data.columns:
            filtered_data = filtered_data.drop(columns=['SpacingBetweenSlices'])

        # Apply Z-score normalization on numerical features
        numerical_features = filtered_data.select_dtypes(include=['number'])
        features = numerical_features.values

        # Normalize the features using StandardScaler
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(features)

        # Apply UMAP
        reducer = umap.UMAP(n_components=2, random_state=42)
        umap_results = reducer.fit_transform(normalized_features)

        filtered_data['UMAP_1'] = umap_results[:, 0]
        filtered_data['UMAP_2'] = umap_results[:, 1]

        method_key = os.path.basename(csv_path).split('_')[1].lower()
        method = method_labels.get(method_key, 'Unknown Method')

        # Create 1x2 subplot
        fig, axes = plt.subplots(1, 2, figsize=(20, 8), gridspec_kw={'width_ratios': [1.2, 1.5]})

        # Plot ROI-based UMAP
        scatter_roi = axes[0].scatter(umap_results[:, 0], umap_results[:, 1], c=filtered_data["ROI_numeric"], cmap='viridis', alpha=0.5)
        roi_labels = list(roi_color_mapping.keys())
        handles, _ = scatter_roi.legend_elements()
        axes[0].legend(handles, roi_labels, title="ROI", loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=4, frameon=False)
        axes[0].grid(True)
        axes[0].set_title(f"UMAP {scanner} {method} - ROI Colored")

        # Plot Dose-based UMAP
        scatter_dose = axes[1].scatter(umap_results[:, 0], umap_results[:, 1], c=filtered_data["dose_numeric"], cmap='coolwarm', alpha=0.5, marker='o')

        # Ensure legend displays ordered dose labels
        sorted_handles_labels = sorted(zip(scatter_dose.legend_elements()[0], unique_labels), key=lambda x: float(x[1].replace("mGy", "")))
        sorted_handles, sorted_labels = zip(*sorted_handles_labels)

        cbar = plt.colorbar(scatter_dose, ax=axes[1])
        cbar.set_label('Dose (mGy)', rotation=270, labelpad=15)

        axes[1].legend(sorted_handles, sorted_labels, title="Dose", loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=5, frameon=False)
        axes[1].grid(True)
        axes[1].set_title(f"UMAP {scanner} {method} - Dose Colored")

        plt.show()

        # Save the subplot
        output_file_subplots = os.path.join(scanner_dir, f"{os.path.splitext(os.path.basename(csv_path))[0]}_umap_plot_{scanner}_roi_and_dose.png")
        plt.savefig(output_file_subplots, bbox_inches='tight')
        print(f"UMAP subplots (ROI and Dose) saved to: {output_file_subplots}")
        plt.close()


def main():
    files_dir = '/mnt/nas7/data/maria/final_features/small_roi'
    output_dir = '/mnt/nas7/data/maria/final_features/umap_results_dose'

    csv_path = [
        f'{files_dir}/features_pyradiomics_full.csv',
        f'{files_dir}/features_cnn_full.csv',
        f'{files_dir}/features_swinunetr_full.csv',
        f'{files_dir}/features_ct-fm_full.csv'
    ]

    for path in csv_path:
        apply_umap(path, output_dir=output_dir)

if __name__ == '__main__':
    main()
