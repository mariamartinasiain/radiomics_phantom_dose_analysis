import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib as mpl
import os

def create_custom_legend(labels, colors, title, filename, output_path):
    """Create and save a custom legend with circle markers and bold title."""
    # Create circular legend markers
    handles = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor=color, markeredgecolor='none', markersize=11, label=label)
        for label, color in zip(labels, colors)
    ]

    fig, ax = plt.subplots(figsize=(2, len(labels) * 0.5 + 0.5))
    ax.axis('off')

    legend = ax.legend(
        handles=handles,
        title=title,
        title_fontsize='large',
        loc='center left',
        bbox_to_anchor=(0, 0.5),
        fontsize='large',
        frameon=False,
        ncol=1
    )

    # Make legend title bold
    legend.get_title().set_fontweight('bold')
    legend.get_title().set_fontsize(14) 
    legend.get_title().set_position((0, 1.3))  # (x, y), where y > 1 moves it up a bit


    plt.tight_layout()
    full_path = os.path.join(output_path, filename)
    plt.savefig(full_path, bbox_inches='tight', dpi=300)
    plt.close()

output_path = '/mnt/nas7/data/maria/final_features/final_features_complete/legends'
os.makedirs(output_path, exist_ok=True)

# === First legend: ROI classes using viridis-style colors ===
roi_labels = ['Cyst', 'Hemangioma', 'Metastasis', 'Normal']
roi_colors = ['#440154', '#31688e', '#35b779', '#fde725']
create_custom_legend(roi_labels, roi_colors, 'ROI', 'legend_roi.png', output_path)

# === Second legend: Manufacturers using tab10-like custom colors ===
manufacturer_labels = ['GE MEDICAL SYSTEMS', 'PHILIPS', 'SIEMENS', 'TOSHIBA']
manufacturer_colors = ['#1f77b4', '#d62728', '#e377c2', '#17becf']
create_custom_legend(manufacturer_labels, manufacturer_colors, 'Manufacturer', 'legend_manufacturer.png', output_path)


# Titles to display
titles = ['PyRadiomics', 'Shallow CNN', 'SwinUNETR', 'CT-FM']

# Create the figure
fig, ax = plt.subplots(figsize=(4, 2.5))  # Adjust size as needed
ax.axis('off')

# Plot each title in a new line
for i, title in enumerate(titles):
    ax.text(0.5, 1 - i * 0.25, title, fontsize=20, ha='center', va='center', fontweight='bold')

# Adjust layout and save
plt.tight_layout()
plt.savefig(os.path.join(output_path, 'model_titles.png'), dpi=300, bbox_inches='tight')
plt.close()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_accuracy_matrix(file_path, method_name):
    # Load the data
    df = pd.read_csv(file_path)

    # Pivot the data into a matrix form (Training dose vs Testing dose)
    accuracy_matrix = df.pivot(index='test_dose', columns='train_dose', values='validation_accuracy')

    # Create the plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(accuracy_matrix, annot=True, cmap="Blues", vmin=0.70, vmax=1.0, fmt='.3f', cbar_kws={'label': 'Accuracy'}, square=True)


    # Add titles and labels
    plt.title(f"{method_name}", fontsize=14, fontweight='bold')
    plt.xlabel('Training Dose (mGy)', fontsize=11)
    plt.ylabel('Testing Dose (mGy)', fontsize=11)
    
    # Save the plot to a file
    output_path = '/mnt/nas7/data/maria/final_features/final_features_complete/legends'
    plt.tight_layout()
    plt.savefig(f"{output_path}/accuracy_matrix_{method_name.lower()}.png", dpi=300, bbox_inches='tight')
    plt.close()

# Define file paths and method names for the four methods
files_and_methods = [
    ('/mnt/nas7/data/maria/final_features/final_features_complete/roi_classification/four_rois/detailed_results_pyradiomics_mlp.csv', 'PyRadiomics'),
    ('/mnt/nas7/data/maria/final_features/final_features_complete/roi_classification/four_rois/detailed_results_cnn_mlp.csv', 'Shallow CNN'),
    ('/mnt/nas7/data/maria/final_features/final_features_complete/roi_classification/four_rois/detailed_results_swinunetr_mlp.csv', 'SwinUNETR'),
    ('/mnt/nas7/data/maria/final_features/final_features_complete/roi_classification/four_rois/detailed_results_ct-fm_mlp.csv', 'CT-FM')
]

# Loop over the file-method pairs and generate plots
for file_path, method_name in files_and_methods:
    plot_accuracy_matrix(file_path, method_name)


