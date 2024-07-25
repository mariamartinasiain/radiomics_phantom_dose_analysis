import pandas as pd
import numpy as np
from neuroCombat import neuroCombat
import ast

# Charger le CSV
df = pd.read_csv('random_contrast_features.csv')

# Convertir la chaîne de caractères en liste pour les deepfeatures
df['deepfeatures'] = df['deepfeatures'].apply(ast.literal_eval)

# Créer la matrice de données
data = np.array(df['deepfeatures'].tolist())

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
df['deepfeatures'] = combat_data.tolist()

# Sauvegarder dans un nouveau CSV
df.to_csv('harmonized_data.csv', index=False)

# Optionnel : Vérifier que toutes les colonnes sont présentes
print(df.columns)