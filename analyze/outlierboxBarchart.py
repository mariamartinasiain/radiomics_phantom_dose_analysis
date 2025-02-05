# The New BoxPlot and BarChart asked by Adrien

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sns.set_style()
fontsize = 16
fontsize2 = 24
figsize = (12, 8)
figsize2 = (6, 6)

files_dir = '/home/reza/radiomics_phantom/results'
suffix = ''
files_dir = '/home/reza/radiomics_phantom/results_combat'
suffix = '\nCombat'
mode = 'roi_small'#'scanner'#


files = {
    f'Pyradiomics{suffix}': f'{files_dir}/results_{mode}_9999_10_features_pyradiomics_full.csv',
    f'Shallow CNN{suffix}': f'{files_dir}/results_{mode}_9999_10_features_oscar_full.csv',
    f'SwinUNETR{suffix}': f'{files_dir}/results_{mode}_9999_10_features_swinunetr_full.csv',
    f'SwinUNETR\nContrastive{suffix}': f'{files_dir}/results_{mode}_9999_10_features_swinunetr_contrastive_full_loso.csv'
    }

# Read the data from the csv files
data = {}
for key in files.keys():
    data[key] = pd.read_csv(files[key])

# Create a boxplot
fig, ax = plt.subplots(figsize=figsize2)
for i, key in enumerate(data.keys()):
    dataframe = data[key]
    # print(dataframe['12'])
    plt.boxplot(dataframe['12'], positions=[i], showfliers=True)
plt.xlabel('')
plt.ylabel('Accuracy of\nleave-one-scanner-out', fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.ylim(0.3, 1.02)
# ax.set_title(title)
# ax.set_ylabel(y_label)
plt.xticks(fontsize=fontsize)
# plt.yticks(np.linspace(0.8, 1, 11), fontsize=fontsize)
#ax.set_xlabel('Feature Extraction Method')
# Change the xtiks to the keys in the data
plt.xticks(range(len(data.keys())), data.keys(), fontsize=fontsize)
# Make the xticks vertical
plt.xticks(rotation=90)
# Remove the empty space between the boxes
plt.grid(axis='y')
plt.tight_layout()
fig.savefig('OutLierBoxPlot.png')
plt.show()
# Read the data from the csv files
data = {}
for key in files.keys():
    data[key] = pd.read_csv(files[key])

# Create a boxplot
fig, ax = plt.subplots(figsize=figsize)
for i, key in enumerate(data.keys()):
    dataframe = data[key]
    # print(dataframe['12'])
    means =  dataframe['1'].mean()               # Mean of each column
    std_error = dataframe['1'].std()/np.sqrt(len(dataframe['1']))             # Standard deviation (variance)
    plt.bar([i], [means])  # Barplot without internal error bars
    # Add custom error bars using plt.errorbar
    plt.errorbar(x=[i], y=means, yerr=std_error, fmt='.', color='black', capsize=3)
plt.xlabel('')
plt.ylabel('Mean accuracy of\ntraining on one scanner', fontsize=fontsize2)
# ax.set_title(title)
# ax.set_ylabel(y_label)
# plt.yticks(np.linspace(0.8, 1, 11), fontsize=fontsize)
#ax.set_xlabel('Feature Extraction Method')
# Change the xtiks to the keys in the data
plt.xticks(range(len(data.keys())), data.keys(), fontsize=fontsize2)
# Make the xticks vertical
plt.xticks(rotation=90)
plt.xticks(fontsize=fontsize2)
plt.yticks(fontsize=fontsize2-4)
# Remove the empty space between the boxes
plt.ylim(0, 1.0)
plt.grid(axis='y')
plt.suptitle('')
plt.tight_layout()
plt.show()
fig.savefig('OutLierBoxPlot1.png')