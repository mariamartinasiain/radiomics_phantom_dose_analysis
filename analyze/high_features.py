import pandas as pd
import numpy as np

def load_feature_set(csv_path):
    """
    Load the feature set CSV and convert deep features from strings to arrays.
    """
    feature_set_df = pd.read_csv(csv_path)
    
    # Convert the deepfeatures column to numerical arrays
    feature_set_df['deepfeatures'] = feature_set_df['deepfeatures'].apply(lambda x: np.fromstring(x.strip("[]"), sep=','))
    
    return feature_set_df

def load_icc_values(icc_csv_path):
    """
    Load the ICC values CSV and sort features by ICC value.
    """
    icc_values_df = pd.read_csv(icc_csv_path)
    icc_values_sorted = icc_values_df.sort_values(by='ICC', ascending=False)
    
    return icc_values_sorted

def select_top_features(icc_values_df, percentage=0.9):
    """
    Select the top percentage of features based on ICC values.
    """
    top_count = int(len(icc_values_df) * percentage)
    top_features = icc_values_df.head(top_count)['Feature'].tolist()
    
    return top_features

def filter_features(deep_features, top_features):
    """
    Filter the deep features to keep only the top features.
    """
    # Select indices of the top features
    feature_indices = [int(f.split('_')[1]) for f in top_features]
    
    # Filter features
    filtered_features = deep_features[feature_indices]
    return filtered_features

def save_filtered_features(feature_set_df, filtered_features_list, output_path):
    """
    Save the filtered feature set to a CSV file.
    """
    # Convert filtered features list back to array format
    feature_set_df['deepfeatures'] = [list(features) for features in filtered_features_list]
    
    feature_set_df.to_csv(output_path, index=False)
    print(f"Filtered features saved to {output_path}")

def filter(feature_set_path, icc_values_path, output_path):
    feature_set_df = load_feature_set(feature_set_path)
    icc_values_df = load_icc_values(icc_values_path)
    top_features = select_top_features(icc_values_df, percentage=0.9)
    
    # Filter each row's deep features
    filtered_features_list = []
    for _, row in feature_set_df.iterrows():
        filtered_features = filter_features(row['deepfeatures'], top_features)
        filtered_features_list.append(filtered_features)
    
    # Save the filtered features to the output path
    save_filtered_features(feature_set_df, filtered_features_list, output_path)

    # Print the number of new features
    num_features = len(top_features)
    print(f"Number of features after filtering: {num_features}")

def main():
    features_paths = ["2combat_features_oscar_full.csv", "2combat_features_pyradiomics_full.csv", "2combat_features_swinunetr_full.csv"]

    for feature_set_path in features_paths:
        icc_values_path = f'nicc_{feature_set_path}' 
        output_path = f'filtered_{feature_set_path}'

        filter(feature_set_path, icc_values_path, output_path)

if __name__ == '__main__':
    main()
