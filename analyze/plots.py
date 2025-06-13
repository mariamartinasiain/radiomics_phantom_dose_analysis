import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import matplotlib.patches as patches



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

output_path = '/mnt/nas7/data/maria/final_features/plots'
os.makedirs(output_path, exist_ok=True)

# === First legend: ROI classes using viridis-style colors ===
roi_labels = ['Cyst', 'Hemangioma', 'Metastasis', 'Normal']
roi_colors = ['#440154', '#31688e', '#35b779', '#fde725']
create_custom_legend(roi_labels, roi_colors, 'Liver tissue class', 'legend_roi.png', output_path)

# === Second legend: Manufacturers using tab10-like custom colors ===
manufacturer_labels = ['GE Medical Systems', 'Philips', 'Siemens', 'Toshiba']
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
    output_path = '/mnt/nas7/data/maria/final_features/plots'
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

output_path = '/mnt/nas7/data/maria/final_features/final_features_complete/plots'
os.makedirs(output_path, exist_ok=True)

# === First legend: ROI classes using viridis-style colors ===
roi_labels = ['Bladder', 'Bone', 'Brain', 'Kidneys', 'Liver', 'Lungs']
tab10 = plt.get_cmap('tab10')
roi_colors = [tab10(i) for i in range(len(roi_labels))]
create_custom_legend(roi_labels, roi_colors, 'Organs', 'legend_org.png', output_path)





output_dir = "/mnt/nas7/data/maria/final_features/histograms"
os.makedirs(output_dir, exist_ok=True)

images = {
    "14mGy": "/mnt/nas7/data/reza/registered_dataset_all_doses/A1_174008_691000_SOMATOM_Definition_Edge_ID3_Harmonized_14mGy_IR_NrFiles_343.nii.gz",
    "10mGy": "/mnt/nas7/data/reza/registered_dataset_all_doses/A1_174008_691000_SOMATOM_Definition_Edge_ID23_Harmonized_10mGy_IR_NrFiles_343.nii.gz",
    "6mGy":  "/mnt/nas7/data/reza/registered_dataset_all_doses/A1_174008_691000_SOMATOM_Definition_Edge_ID43_Harmonized_6mGy_IR_NrFiles_343.nii.gz",
    "3mGy":  "/mnt/nas7/data/reza/registered_dataset_all_doses/A1_174008_691000_SOMATOM_Definition_Edge_ID63_Harmonized_3mGy_IR_NrFiles_343.nii.gz",
    "1mGy":  "/mnt/nas7/data/reza/registered_dataset_all_doses/A1_174008_691000_SOMATOM_Definition_Edge_ID83_Harmonized_1mGy_IR_NrFiles_343.nii.gz"
}

for label, path in images.items():
    img = nib.load(path)
    data = img.get_fdata()
    #data = data[data > -1000]  # remove background / air

    plt.figure(figsize=(8, 5))
    plt.hist(data.flatten(), bins=200, density=True, alpha=0.75, color='steelblue', histtype='stepfilled')
    plt.xlabel("Intensity (HU)")
    plt.ylabel("Relative Density")
    plt.title(f"Histograma CT - {label}")
    plt.ylim(0, 0.006) 
    plt.grid(True)
    plt.tight_layout()

    # Guardar imagen
    save_path = os.path.join(output_dir, f"histogram_{label}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()


# Crear máscaras para intensidades en torno a -100 y -200 HU
    mask_neg100 = (data >= -200) & (data <= -100)

    # Tomar una slice central
    slice_idx = data.shape[2] // 2
    slice_img = data[:, :, slice_idx]
    slice_mask_100 = mask_neg100[:, :, slice_idx]


    # Visualización y guardado
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(slice_img, cmap='gray', vmin=-300, vmax=300)
    plt.title(f"{label} - Slice Central")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(slice_img, cmap='gray', vmin=-300, vmax=300)
    plt.imshow(slice_mask_100, cmap='viridis', alpha=0.7)
    plt.title(f"{label} - Máscaras")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"masks_overlay_{label}.png"), dpi=300)
    plt.close()


import matplotlib.pyplot as plt
import numpy as np
import os

methods = ['PyRadiomics', 'Shallow CNN', 'SwinUNETR', 'CT-FM']
means = [0.9205, 0.9407, 0.8178, 0.9650]
stds = [0.0293, 0.0172, 0.0350, 0.0190]

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # azul, naranja, verde, rojo

x = np.arange(len(methods))
fig, ax = plt.subplots(figsize=(8, 5))

plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 14,
    'axes.labelsize': 11,
    'xtick.labelsize': 13,
    'ytick.labelsize': 12,
    'legend.fontsize': 15
})

bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors)

ax.set_ylabel('Mean accuracy')
#ax.set_title('10-fold CV Accuracy for Organ Classification')
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.set_ylim(0.7, 1.0)
ax.grid(True, axis='y', linestyle='--', alpha=0.7)

'''
# Mostrar valores a la derecha de la barra de error
for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
    ax.annotate(f'{mean:.3f}', 
                xy=(bar.get_x() + bar.get_width() / 2 + 0.1, mean + std / 2), 
                ha='left', va='center')
'''

plt.tight_layout()

output_path = '/mnt/nas7/data/maria/final_features/plots'
os.makedirs(output_path, exist_ok=True)

output_file = os.path.join(output_path, 'cv_accuracy_plot.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight')

plt.show()
