import pandas as pd
import numpy as np
from pingouin import intraclass_corr

def auto_detect_and_calculate_icc(csv_path, roi_column='ROI', series_column='SeriesDescription' ,feature_column='deepfeatures'):
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
    
    results = []
    for feature in feature_columns:
        icc_data = data[[roi_column, series_column, feature]].dropna()
        icc_data.columns = ['raters', 'targets', 'ratings']
        try:
            icc = intraclass_corr(data=icc_data, raters='raters', targets='targets', ratings='ratings').set_index('Type').at['ICC3k', 'ICC']
            results.append({'Feature': feature, 'ICC': icc})
        except Exception as e:
            results.append({'Feature': feature, 'ICC': np.nan})

    return pd.DataFrame(results)





def main():
    csv_path = '../random_contrast_features.csv'
    icc_results = auto_detect_and_calculate_icc(csv_path)

    icc_results_sorted = icc_results.sort_values(by='ICC', ascending=False)
    print(icc_results_sorted)

    output_path = 'random_contrast_features_icc_results.csv'
    icc_results_sorted.to_csv(output_path, index=False)
    print(f"Les résultats ICC sont enregistrés dans {output_path}")

if __name__ == '__main__':
    main()