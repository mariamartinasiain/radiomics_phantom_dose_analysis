import numpy as np
import pandas as pd

def to_avg():
    filepath = "../../all_dataset_features/swin_deepfeatures.csv"
    data = pd.read_csv(filepath)
    print("we have",len(data),"rows")
    features = data.drop(columns=['StudyInstanceUID', 'SeriesNumber', 'SeriesDescription', 'ROI','ManufacturerModelName','Manufacturer','SliceThickness','SpacingBetweenSlices'],errors='ignore')
    if features.columns[0] == 'deepfeatures':
            features = features['deepfeatures'].apply(eval).apply(pd.Series)
            
    print("we have",len(features.columns),"features")
    reshaped_features = features.values.reshape((-1,768,4))
    averaged_data = reshaped_features.mean(axis=2)

    print("average data shape",averaged_data.shape)

    data['deepfeatures'] = averaged_data.tolist()

    data.to_csv("averaged_swin_features.csv", index=False)
    
def deepfeatures_to_pca():
    from sklearn.decomposition import PCA
    filepath = "../../all_dataset_features/averaged_swin_deepfeatures.csv"
    data = pd.read_csv(filepath)
    print("we have",len(data),"rows")
    features = data.drop(columns=['StudyInstanceUID', 'SeriesNumber', 'SeriesDescription', 'ROI','ManufacturerModelName','Manufacturer','SliceThickness','SpacingBetweenSlices'],errors='ignore')
    if features.columns[0] == 'deepfeatures':
            features = features['deepfeatures'].apply(eval).apply(pd.Series)
            
    print("we have",len(features.columns),"features")
    pca = PCA(n_components=20)
    pca_data = pca.fit_transform(features)
    print("pca data shape",pca_data.shape)
    data['deepfeatures'] = pca_data.tolist()

    data.to_csv("pca_swin_features.csv", index=False)
    
def radiomics_to_pca():
    from sklearn.decomposition import PCA
    filepath = "../../all_dataset_features/pyradiomics_features.csv"
    data = pd.read_csv(filepath)
    print("we have",len(data),"rows")
    features = data.drop(columns=['StudyInstanceUID', 'SeriesNumber', 'SeriesDescription', 'ROI','ManufacturerModelName','Manufacturer','SliceThickness','SpacingBetweenSlices'],errors='ignore')
    print("we have",len(features.columns),"features")
    pca = PCA(n_components=20)
    pca_data = pca.fit_transform(features)
    print("pca data shape",pca_data.shape)
    data['deepfeatures'] = pca_data.tolist()

    data.to_csv("pca_pyradiomics_features.csv", index=False)
    
if __name__ == '__main__':
    radiomics_to_pca()