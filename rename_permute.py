import os
import json

# Load the mapping from the JSON file
with open('name_correspondance.json', 'r') as f:
    name_mapping = json.load(f)

# Directory where the files are located
directory = 'uncompress_cropped/'

# List to hold the temporary names in order
temp_names = []

# First pass: rename all files to a temporary name using an index
for index, current_name in enumerate(name_mapping.keys()):
    current_path = os.path.join(directory, current_name)
    temp_name = f'temp_{index}'  # Generate a temporary name using the index
    temp_path = os.path.join(directory, temp_name)
    
    if os.path.exists(current_path):
        os.rename(current_path, temp_path)
        temp_names.append(temp_name)  # Store the temporary name

# Second pass: rename all temporary names to the desired names
for temp_name in temp_names:
    index = int(temp_name.split('_')[1])
    current_name = list(name_mapping.keys())[index]
    desired_name = name_mapping[current_name]
    temp_path = os.path.join(directory, temp_name)
    desired_path = os.path.join(directory, desired_name)
    
    if os.path.exists(temp_path):
        os.rename(temp_path, desired_path)
        print(f'Renamed: {current_name} to {desired_name}')

print('Renaming completed.')
