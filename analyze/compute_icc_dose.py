import os
import pandas as pd
import numpy as np
from pingouin import intraclass_corr
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Define all scanners
scanners = ["A1", "A2", "B1", "B2", "G1", "G2", "C1", "H2", "D1", "E2", "F1", "E1", "H1"]

# Define all scanners
scanners = ["A1", "A2", "B1", "B2", "G1", "G2", "C1", "H2", "D1", "E2", "F1", "E1", "H1"]

def extract_feature_type(csv_path):
    """Extracts feature type from the filename (Radiomics, CNN, or SwinUNETR)."""
    if "pyradiomics" in csv_path.lower():
        return "Radiomics"
    elif "cnn" in csv_path.lower():
        return "CNN"
    elif "swinunetr" in csv_path.lower():
        return "SwinUNETR"
    return "Unknown"

def calculate_icc(csv_path, roi_column='ROI', series_column='SeriesDescription', feature_column='deepfeatures'):
    data = pd.read_csv(csv_path)
    feature_type = extract_feature_type(csv_path)

    # Extract scanner and dose from SeriesDescription
    data[['Pos0', 'Pos2']] = data['SeriesDescription'].str.split('_', expand=True)[[0, 2]]


    summary = []
    icc_results_per_scanner = {}

    #output_dir = '/mnt/nas7/data/maria/final_features/icc_results_dose'
    output_dir = '/mnt/nas7/data/maria/final_features/icc_results_dose/six_rois'
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for scanner in scanners:
        # Create a directory for each scanner
        scanner_dir = os.path.join(output_dir, f'icc_scanner_{scanner}')
        os.makedirs(scanner_dir, exist_ok=True)

        print(f'Processing scanner: {scanner}')
        filtered_data = data[data['Pos0'] == scanner].copy()
        filtered_data['SeriesDescription'] = filtered_data['Pos2']  # Keep just the doses
        filtered_data.drop(columns=['Pos0', 'Pos2'], inplace=True)
        num_cases = len(filtered_data)
        print(f"Number of cases: {num_cases}")

        # Convert deep features if necessary
        if feature_column in filtered_data.columns and filtered_data[feature_column].dtype == 'object':

            filtered_data[feature_column] = filtered_data[feature_column].apply(lambda x: np.fromstring(x.strip("[]"), sep=','))
            max_len = filtered_data[feature_column].apply(len).max()
            feature_df = pd.DataFrame(filtered_data[feature_column].tolist(), index=filtered_data.index)
            feature_df.columns = [f"feature_{i}" for i in range(max_len)]
            filtered_data = pd.concat([filtered_data.drop(columns=[feature_column]), feature_df], axis=1)

        feature_columns = filtered_data.select_dtypes(include=[np.number]).columns.tolist()
        feature_columns = [col for col in feature_columns if col not in [roi_column, series_column]]

        # Normalize the features (Z-score normalization)
        scaler = StandardScaler()
        filtered_data[feature_columns] = scaler.fit_transform(filtered_data[feature_columns])

        # Rename duplicates for .mat file export
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
        # Rename duplicates for .mat file export
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

        roi_mapping = {roi: i for i, roi in enumerate(filtered_data["ROI"].unique(), start=1)}
        filtered_data = pd.concat([filtered_data, filtered_data["ROI"].map(roi_mapping).rename("ROI_numerical")], axis=1)
        filtered_data.columns = [col[:28] for col in filtered_data.columns]  # Truncate for .mat file format

        try:
            filtered_data = filtered_data.drop(columns=['Unnamed: 0'])
        except KeyError:
            pass

        filtered_data.columns = rename_duplicates(filtered_data.columns)
        filtered_data.columns = rename_duplicates(filtered_data.columns)

        # Compute ICC
        results = []
        for feature in filtered_data.columns:
            icc_data = filtered_data[[series_column, roi_column, feature]].dropna()
            icc_data.columns = ['raters', 'targets', 'ratings']

            try:
                icc_result = intraclass_corr(data=icc_data, raters='raters', targets='targets', ratings='ratings')
                icc = max(0, icc_result.set_index('Type').at['ICC3k', 'ICC'])  # Clip negative ICC values to zero
                results.append({'Feature': feature, 'ICC': icc})

            except Exception as e:
                results.append({'Feature': feature, 'ICC': np.nan})

        icc_results_sorted = pd.DataFrame(results).sort_values(by='ICC', ascending=False)

        # Define a list of values to be removed
        unwanted_values = [
            "SpacingBetweenSlices", "SliceThickness", "ManufacturerModelName", 
            "Manufacturer", "StudyInstanceUID", "SeriesNumber", 
            "SeriesDescription", "ROI"
        ]

        # Filter out rows that contain unwanted values
        icc_results_sorted = icc_results_sorted[~icc_results_sorted['Feature'].isin(unwanted_values)]

        # Drop rows that contain NaN values
        icc_results_sorted = icc_results_sorted.dropna()

        # Save the results
        base_filename = os.path.basename(csv_path).replace('.csv', '')
        icc_results_sorted.to_csv(os.path.join(scanner_dir, f'icc_dose_{base_filename}_{scanner}.csv'), index=False)

        icc_results_per_scanner[(csv_path, scanner)] = icc_results_sorted
        summary.append({'Scanner': scanner, 'Num_Cases': num_cases, 'Num_Features': len(icc_results_sorted)})


    return summary, icc_results_per_scanner

def main():
    #files_dir = '/mnt/nas7/data/maria/final_features/small_roi'
    files_dir = '/mnt/nas7/data/maria/final_features'
    output_dir = '/mnt/nas7/data/maria/final_features/icc_results_dose/six_rois'
    os.makedirs(output_dir, exist_ok=True)

    csv_paths = [
        #f'{files_dir}/features_pyradiomics_full.csv',
        #f'{files_dir}/features_cnn_full.csv',
        #f'{files_dir}/features_swinunetr_full.csv',
        f'{files_dir}/features_swinunetr_reversed.csv'
    ]

    for path in csv_paths:
        summaries, icc_results_per_scanner = calculate_icc(path)


        # Save final summary
        summary_df = pd.DataFrame(summaries)
        summary_df.to_csv(f'{output_dir}/icc_summary_{os.path.basename(path)}', index=False)

if __name__ == '__main__':
    main()



