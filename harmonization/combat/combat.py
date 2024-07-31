import pandas as pd
import numpy as np
from neuroCombat import neuroCombat
import ast


fname = "features_oscar_full"
df = pd.read_csv(f'{fname}.csv')

# Convertir la chaîne de caractères en liste pour les deepfeatures
#df['deepfeatures'] = df['deepfeatures'].apply(ast.literal_eval)

#données
data = df.drop(columns=['StudyInstanceUID', 'SeriesNumber', 'SeriesDescription', 'ROI','ManufacturerModelName','Manufacturer','SliceThickness','SpacingBetweenSlices'],errors='ignore')
if data.columns[0] == 'deepfeatures':
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
                          batch_col='ROI')['data'].T

if exit:
    df['deepfeatures'] = combat_data.tolist()
else:
    df['deepfeatures'] = combat_data.tolist()

df.to_csv(f'combat_{fname}.csv', index=False)
print(df.columns)