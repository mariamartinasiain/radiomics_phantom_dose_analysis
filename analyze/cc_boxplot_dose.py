import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Function to load data
def load_data(file_path):
    return pd.read_csv(file_path)

# Function to create boxplots for each scanner and method
def create_boxplots(ax, method_data, scanners, title):
    # Create a boxplot for each scanner
    data_to_plot = [method_data[scanner]['ICC'] for scanner in scanners if scanner in method_data]

    custom_labels = ['Pyradiomics', 'Shallow CNN', 'SwinUNETR']

    box = ax.boxplot(data_to_plot, labels=custom_labels, showfliers=False)

    # Add N above each box (but below the title)
    for i, method in enumerate(method_data.keys()):
        n = len(method_data[method]['ICC']) - 1     # Without taking into account "ROI_numerical"
        max_val = max(method_data[method]['ICC'])  # Get the max ICC value in the boxplot
        ax.text(i + 1, max_val + 0.01, f'N={n}', ha='center', va='bottom', fontsize=12)


    ax.set_title(f'{title}', fontsize=14, fontweight='bold', pad=25) 
    ax.set_ylabel('ICC', fontsize=12)
    ax.set_ylim(0.3, 1)
    ax.grid(True, axis='y', linestyle="solid", linewidth=0.5) 
    ax.set_yticks(np.arange(0.3, 1.05, 0.05))  # Grid every 0.05
    ax.tick_params(axis='x', labelsize=14)

def main():
    files_dir = '/mnt/nas7/data/maria/final_features/icc_results_dose'
    methods = ['pyradiomics', 'cnn', 'swinunetr']
    scanners = ['A1', 'A2', 'B1', 'B2']

    all_data = {scanner: {} for scanner in scanners}

    # Load data for each method and scanner
    for method in methods:
        for scanner in scanners:
            file_path = f'{files_dir}/icc_dose_features_{method}_full_{scanner}.csv'
            icc_data = load_data(file_path)
            all_data[scanner][method] = icc_data

    print(all_data)

    # Create a figure and generate boxplots for each scanner
    output_dir = '/mnt/nas7/data/maria/final_features/icc_results_dose/icc_boxplot'
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

    for scanner in scanners:
        fig, ax = plt.subplots(figsize=(10, 8))
        create_boxplots(ax, all_data[scanner], methods, f'Scanner {scanner}')
        output_path = os.path.join(output_dir, f'{scanner}_boxplot_dose_comparison.png')
        fig.savefig(output_path, dpi=300)  # Save the figure as a PNG file

        print(f'Plot for {scanner} saved.')

if __name__ == "__main__":
    main()
