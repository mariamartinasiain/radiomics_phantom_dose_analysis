import pandas as pd
import numpy as np
import os
import ast  # Safer than eval

output_dir = '/mnt/nas7/data/maria/final_features/'
os.makedirs(output_dir, exist_ok=True)

# Load the CSV containing the flattened features
#csv_file = '/mnt/nas7/data/maria/final_features/features_swinunetr_full.csv'  # Replace with your actual CSV file
csv_file = '/home/reza/radiomics_phantom/final_features_doses/features_swin.csv'

df = pd.read_csv(csv_file)

# Function to reverse the flattening
def reverse_flatten(flattened_features):
    flattened_features = np.array(flattened_features)
    reshaped_features = flattened_features.reshape(768, 2, 2)
    averaged_features = reshaped_features.mean(axis=(1, 2))  # Average over 2x2
    return averaged_features.tolist()  # Convert back to a list for CSV storage

# Apply to each row and replace the 'deepfeatures' column
df['deepfeatures'] = df['deepfeatures'].apply(lambda x: reverse_flatten(ast.literal_eval(x)))

# Save the updated DataFrame with the corrected deepfeatures
output_file = os.path.join(output_dir, 'features_swinunetr_reversed.csv')
df.to_csv(output_file, index=False)

print(f"Saved corrected features to {output_file}")
