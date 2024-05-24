import pandas as pd
import numpy as np
import ast

# Load the dataset
df = pd.read_csv('../../all_dataset_features/features_swinunetr_full.csv')

# Function to parse and average the deepfeatures column
def average_features(row):
    # Safely evaluate the string to get the list of features
    features_list = ast.literal_eval(row)
    # Convert the list to a numpy array
    features = np.array(features_list)
    # Reshape and average every 768 elements
    averaged_features = features.reshape(-1, 768).mean(axis=0)
    return ','.join(map(str, averaged_features))

# Apply the function to the deepfeatures column
df['deepfeatures'] = df['deepfeatures'].apply(average_features)

# Save the new DataFrame to a CSV file
df.to_csv('averaged_correct_contrastive_deepfeatures.csv', index=False)

print("Averaged features saved to 'averaged_features_swinunetr_full.csv'")
