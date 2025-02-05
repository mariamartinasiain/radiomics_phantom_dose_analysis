import pandas as pd
import numpy as np
from neuroCombat import neuroCombat
import ast

def features_to_numpy(features):
    try:
        features_array = np.zeros([len(features), len(features['deepfeatures'].iloc[0])])
    except:
        return np.array(features)
    for i, row in enumerate(features['deepfeatures']):
        features_array[i] = row
    return features_array

#fnames = ["features_oscar_full", "features_pyradiomics_full", "features_swinunetr_full"]
files_dir = '/home/reza/radiomics_phantom/final_features/small_roi'
fnames = [
    f'{files_dir}/features_pyradiomics_full',
    f'{files_dir}/features_oscar_full',
    f'{files_dir}/features_swinunetr_full',
    f'{files_dir}/features_swinunetr_contrastive_full',
    ]

files_dir = '/home/reza/radiomics_phantom/final_features/small_roi/features_loso'
fnames = [f'{files_dir}/features_liverrandom_contrast_5_15_10_batch_swin_loso_{scanner_id :02d}' for scanner_id in range(13)]
for fname in fnames:
    df = pd.read_csv(f'{fname}.csv')

    # Convertir la chaîne de caractères en liste pour les deepfeatures
    #df['deepfeatures'] = df['deepfeatures'].apply(ast.literal_eval)

    #données
    data = df.drop(columns=['StudyInstanceUID', 'SeriesNumber', 'SeriesDescription', 'ROI','ManufacturerModelName','Manufacturer','SliceThickness','SpacingBetweenSlices'],errors='ignore')
    if data.columns[1] == 'deepfeatures':
        exit = True
        data = df['deepfeatures'].apply(eval).apply(pd.Series)
        #data = np.array(df['deepfeatures'].tolist())
    else:
        exit = False

    #variables de lot (scanner)
    batch = df['SeriesDescription'].apply(lambda x: x[:2])

    #covariables (ROI)
    covars = {'ROI': df['ROI'].values, 'batch': batch.values}
    covars = pd.DataFrame(covars)

    combat_data = neuroCombat(dat=data.T,
                            covars=covars,
                            batch_col='batch')['data'].T

    if exit:
        df['deepfeatures'] = combat_data.tolist()
    else:
        df['deepfeatures'] = combat_data.tolist()

    df.to_csv(f'{fname}_2combat.csv', index=False)
    print(df.columns)