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

'''
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
roi_labels = ['Bladder', 'Bone', 'Brain', 'Kidneys', 'Liver', 'Lungs']
tab10 = plt.get_cmap('tab10')
roi_colors = [tab10(i) for i in range(len(roi_labels))]
create_custom_legend(roi_labels, roi_colors, 'Organs', 'legend_org.png', output_path)



import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os

output_dir = "/mnt/nas7/data/maria/final_features/final_features_complete/histograms"
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



import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

images = {
    "14mGy": "/mnt/nas7/data/reza/registered_dataset_all_doses/A1_174008_691000_SOMATOM_Definition_Edge_ID3_Harmonized_14mGy_IR_NrFiles_343.nii.gz",
    "10mGy": "/mnt/nas7/data/reza/registered_dataset_all_doses/A1_174008_691000_SOMATOM_Definition_Edge_ID23_Harmonized_10mGy_IR_NrFiles_343.nii.gz",
    "6mGy":  "/mnt/nas7/data/reza/registered_dataset_all_doses/A1_174008_691000_SOMATOM_Definition_Edge_ID43_Harmonized_6mGy_IR_NrFiles_343.nii.gz",
    "3mGy":  "/mnt/nas7/data/reza/registered_dataset_all_doses/A1_174008_691000_SOMATOM_Definition_Edge_ID63_Harmonized_3mGy_IR_NrFiles_343.nii.gz",
    "1mGy":  "/mnt/nas7/data/reza/registered_dataset_all_doses/A1_174008_691000_SOMATOM_Definition_Edge_ID83_Harmonized_1mGy_IR_NrFiles_343.nii.gz"
}

output_dir = "/mnt/nas7/data/maria/final_features/final_features_complete/patch_zoom"
os.makedirs(output_dir, exist_ok=True)

def window_level(image, window=400, level=50):
    min_val = level - window / 2
    max_val = level + window / 2
    img_wl = np.clip(image, min_val, max_val)
    img_wl = (img_wl - min_val) / (max_val - min_val)
    return img_wl

zoom_size = 50

def flip_volume(volume, axis=0):
    volume = np.swapaxes(volume, 0, axis)
    volume_flipped = np.zeros_like(volume)
    nslices = volume.shape[0]
    for i in range(nslices):
         volume_flipped[i, ...] = volume[nslices - i - 1, ...]
    volume_flipped = np.swapaxes(volume_flipped, axis, 0)
    return volume_flipped

for label, path in images.items():
    img = nib.load(path)
    data = img.get_fdata()
    affine = img.affine

    data = data.transpose(1, 0, 2)
    data = flip_volume(data, axis=0)

    slice_idx = data.shape[2] // 2
    axial_slice = data[:, :, slice_idx]

    # Aplicar window/level
    axial_wl = window_level(axial_slice, window=400, level=50)

    center_x, center_y = axial_slice.shape[1] // 2, axial_slice.shape[0] // 2
    x_start = center_x - zoom_size // 2 + 25
    y_start = center_y - zoom_size // 2 + 80

    patch = axial_wl[y_start:y_start+zoom_size, x_start:x_start+zoom_size]

    # Calcular coordenadas físicas del patch en 3D
    # Voxel coordinate de la esquina superior izquierda del patch
    voxel_start = np.array([x_start, y_start, slice_idx, 1])  # Añadimos 1 para producto matricial con affine 4x4

    # Convertir a espacio físico (en mm)
    phys_start = affine @ voxel_start

    # Calcular la esquina opuesta (inferior derecha)
    voxel_end = np.array([x_start + zoom_size, y_start + zoom_size, slice_idx, 1])
    phys_end = affine @ voxel_end

    print(f"{label} - Coordenadas voxel patch: start={voxel_start[:3]}, end={voxel_end[:3]}")
    print(f"{label} - Coordenadas físicas patch (mm): start={phys_start[:3]}, end={phys_end[:3]}")

    phys_start_lps = phys_start.copy()
    phys_end_lps = phys_end.copy()

    # Invertir Y (A → P) y Z (S → I)
    phys_start_lps[1] *= -1
    phys_start_lps[2] *= -1
    phys_end_lps[1] *= -1
    phys_end_lps[2] *= -1

    print(f"L-R (X): {phys_start_lps[0]:.2f} mm to {phys_end_lps[0]:.2f} mm")
    print(f"P-A (Y): {phys_start_lps[1]:.2f} mm to {phys_end_lps[1]:.2f} mm")
    print(f"I-S (Z): {phys_start_lps[2]:.2f} mm to {phys_end_lps[2]:.2f} mm")


    import nibabel as nib
    from nibabel.orientations import aff2axcodes

    img = nib.load(path)
    axcodes = aff2axcodes(img.affine)
    print("Orientación de los ejes según el affine:", axcodes)


    # Guardar patch en NIfTI con affine ajustada (opcional)
    patch_3d = patch[:, :, np.newaxis]
    new_origin = affine[:3, :3] @ np.array([x_start, y_start, slice_idx]) + affine[:3, 3]
    patch_affine = affine.copy()
    patch_affine[:3, 3] = new_origin
    patch_img = nib.Nifti1Image(patch_3d, patch_affine)
    out_path = os.path.join(output_dir, f"patch_wl_{label}.nii.gz")
    nib.save(patch_img, out_path)

    # Visualización con recuadro rojo
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f'{label}', fontsize=16)

    axs[0].imshow(axial_wl, cmap='gray')
    rect = patches.Rectangle((x_start, y_start), zoom_size, zoom_size, linewidth=2, edgecolor='red', facecolor='none')
    axs[0].add_patch(rect)
    axs[0].set_title('Slice axial central con WL')
    axs[0].axis('off')

    axs[1].imshow(patch, cmap='gray')
    axs[1].set_title(f'Zoom patch {zoom_size}x{zoom_size} con WL')
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()


'''