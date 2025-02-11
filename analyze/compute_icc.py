import os
import pandas as pd
import numpy as np
from pingouin import intraclass_corr

def auto_detect_and_calculate_icc(csv_path, roi_column='ROI', series_column='SeriesDescription', feature_column='deepfeatures'):
    data = pd.read_csv(csv_path)
    data['SeriesDescription'] = data['SeriesDescription'].apply(lambda x: x.split('_')[0])
    
    if feature_column in data.columns and data[feature_column].dtype == 'object':
        data[feature_column] = data[feature_column].apply(lambda x: np.fromstring(x.strip("[]"), sep=','))
        max_len = data[feature_column].apply(len).max()
        feature_df = pd.DataFrame(data[feature_column].tolist(), index=data.index)
        feature_df.columns = [f"feature_{i}" for i in range(max_len)]
        data = pd.concat([data.drop(columns=[feature_column]), feature_df], axis=1)
    
    feature_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    feature_columns = [col for col in feature_columns if col not in [roi_column, series_column]]
    

    # Import data to Matlab:
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

    from scipy.io import savemat
    roi_mapping = {roi: i for i, roi in enumerate(data["ROI"].unique(), start=1)}
    data["ROI_numerical"] = data["ROI"].map(roi_mapping)
    data.columns = [col[:28] for col in data.columns]  # to be able to save in .mat file format
    try:
        data = data.drop(columns=['Unnamed: 0'])
    except:
        print('Unammed: 0 not found!')
    data.columns = rename_duplicates(data.columns)
    # data.columns = [
    # f'feature_{i:03d}' if len(col) > 30 else col
    # for i, col in enumerate(data.columns)]
    savemat("data.mat", {"dataframe": data.to_dict("list")})
    
    results = []
    #for feature in feature_columns:
    for feature in data.columns:
        icc_data = data[[series_column,roi_column, feature]].dropna()
        icc_data.columns = ['raters', 'targets', 'ratings']
        try:
            icc = intraclass_corr(data=icc_data, raters='raters', targets='targets', ratings='ratings').set_index('Type').at['ICC3k', 'ICC']
            results.append({'Feature': feature, 'ICC': icc})
        except Exception as e:
            results.append({'Feature': feature, 'ICC': np.nan})
    # Remove the elements with ICC nan or inf
    results_filtered = [result for result in results if not np.isnan(result['ICC']) and not np.isinf(result['ICC'])]
    results_garbage = [result for result in results if np.isnan(result['ICC']) or np.isinf(result['ICC'])]

    # ICC is nan for SpacingBetweenSlices, SliceThickness, SeriesNumber
    # Warnings of division by zero are for ManufacturerModelName, Manufacturer, StudyInstanceUID, SeriesDescription, ROI
    return pd.DataFrame(results_filtered)



def main():
    # files_dir = '/home/reza/radiomics_phantom/final_features/small_roi_combat'
    files_dir = '/mnt/nas7/data/maria/final_features/small_roi'
    output_dir = '/mnt/nas7/data/maria/final_features/icc_results'

    csv_path = [
        f'{files_dir}/features_pyradiomics_full.csv',
        f'{files_dir}/features_cnn_full.csv',
        f'{files_dir}/features_swinunetr_full.csv',
        f'{files_dir}/features_swinunetr_contrastive_full.csv',
        f'{files_dir}/features_swinunetr_contrastive_full_loso.csv'
        ]

    for path in csv_path:
        icc_results = auto_detect_and_calculate_icc(path)
    
        icc_results_sorted = icc_results.sort_values(by='ICC', ascending=False)
        print(icc_results_sorted)
        icc_results_sorted.to_csv(f'{output_dir}/nicc_{os.path.basename(path)}', index=False)

if __name__ == '__main__':
    main()