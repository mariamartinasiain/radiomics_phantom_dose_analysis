import pandas as pd
import numpy as np
from neuroCombat import neuroCombat
import ast


fname = "features_pyradiomics_full"
# Charger le CSV
df = pd.read_csv(f'{fname}.csv')



# Convertir la chaîne de caractères en liste pour les deepfeatures
#df['deepfeatures'] = df['deepfeatures'].apply(ast.literal_eval)

# Créer la matrice de données
data = df.drop(columns=['StudyInstanceUID', 'SeriesNumber', 'SeriesDescription', 'ROI','ManufacturerModelName','Manufacturer','SliceThickness','SpacingBetweenSlices'],errors='ignore')
#verifier si features est plutot une liste ou un string d'une liste
if data.columns[0] == 'deepfeatures':
    exit = True
    data = df['deepfeatures'].apply(eval).apply(pd.Series)
    #data = np.array(df['deepfeatures'].tolist())
else:
    exit = False

# Préparer les variables de lot (scanner)
batch = df['SeriesDescription'].apply(lambda x: x[:2])

# Préparer les covariables (ROI)
covars = pd.get_dummies(df['ROI'], prefix='ROI')

# Ajouter la variable de lot aux covariables
covars['batch'] = batch

# Appliquer ComBat
combat_data = neuroCombat(dat=data.T,
                          covars=covars,
                          batch_col='batch')['data'].T

# Remplacer les deepfeatures originaux par les features harmonisés
if exit:
    df['deepfeatures'] = combat_data.tolist()
else:
    #add a new column with the harmonized features
    df['deepfeatures'] = combat_data.tolist()

# Sauvegarder dans un nouveau CSV
df.to_csv(f'combat_{fname}.csv', index=False)

# Optionnel : Vérifier que toutes les colonnes sont présentes
print(df.columns)