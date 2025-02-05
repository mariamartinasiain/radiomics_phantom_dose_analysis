import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data(file_path, method_name):
    """Load data from a CSV file and add a method column."""
    data = pd.read_csv(file_path)
    data['Method'] = method_name
    return data

def create_boxplot(data, value_column, title, y_label, output_path):
    """Create and save a boxplot for a specified value column."""
    fig, ax = plt.subplots(figsize=(8, 8))
    fontsize = 16

    ax = plt.gca()
    ax.text(1, 0.995, 'N=87', color='k', fontsize=fontsize,
                  transform=ax.get_xaxis_transform(), ha='center', va='top')
    ax.text(2, 0.995, 'N=2048', color='k', fontsize=fontsize,
                  transform=ax.get_xaxis_transform(), ha='center', va='top')
    ax.text(3, 0.995, 'N=3072', color='k', fontsize=fontsize,
                  transform=ax.get_xaxis_transform(), ha='center', va='top')
    ax.text(4, 0.995, 'N=3072', color='k', fontsize=fontsize,
                  transform=ax.get_xaxis_transform(), ha='center', va='top')
    
    # Remove outliers
    #data = data[data[value_column] < data[value_column].quantile(0.95)]

    # Create boxplot without outliers
    data.boxplot(column=value_column, by='Method', ax=ax, showfliers=False)
    plt.xlabel('')
    plt.ylabel('')
    # plt.ylim(0.5, 1)
    ax.set_title(title)
    #ax.set_ylabel(y_label)
    plt.xticks(fontsize=fontsize)
    plt.yticks(np.linspace(0.5, 1, 11), fontsize=fontsize)
    #ax.set_xlabel('Feature Extraction Method')
    plt.suptitle('')
    plt.tight_layout()
    plt.show()
    fig.savefig(output_path)

def main():
    # Load data for different methods
    # swin_data = load_data('icc_features_swinunetr_full.csv', ' Swin')
    # finetuneswin_data = load_data('icc_features_paper_contrastive2_F1.csv', ' rois Swin')
    # randomcrop_swin_data = load_data('icc_features_random_contrast_5_6_lowLR_12batch_swin.csv', ' randomcrop Swin')
    # livercrops_swin_data = load_data('nicc_features_liverrandom_contrast_5_15_10batch_swin.csv', ' livercrops Swin')
    
    files_dir = '/home/reza/radiomics_phantom/final_features/small_roi_combat'
    suffix = '\nCombat'
    
    # files_dir = '/home/reza/radiomics_phantom/final_features/small_roi'
    # suffix = ''

    swin_data = load_data(f'{files_dir}/nicc_features_pyradiomics_full.csv', f'Pyradiomics{suffix}')
    finetuneswin_data = load_data(f'{files_dir}/nicc_features_oscar_full.csv', f'Shallow CNN{suffix}') 
    randomcrop_swin_data = load_data(f'{files_dir}/nicc_features_swinunetr_full.csv', f'SwinUNETR{suffix}')
    livercrops_swin_data = load_data(f'{files_dir}/nicc_features_swinunetr_contrastive_full.csv', f'SwinUNETR\nContrastive{suffix}')
    # livercrops_swin_data = load_data(f'{files_dir}/nicc_features_swinunetr_contrastive_full_loso.csv', 'SwinUNETR\nContrastive')
    
    # combat_swin_data = load_data('icc_combat_features_swinunetr_full.csv', 'combat Swin')
    # pyradiomics_data = load_data('icc_features_pyradiomics_full.csv', 'Radiomics')
    # cnn_data = load_data('icc_features_oscar_full.csv', 'cnn')
    # combat_cnn_data = load_data('icc_2combat_features_oscar_full.csv', 'combat cnn')
    # combat_pyradiomics_data = load_data('icc_combat_features_pyradiomics_full.csv', ' combat Radiomics')
    
    # Combine data into a single DataFrame
    data_list = [swin_data, finetuneswin_data, randomcrop_swin_data, livercrops_swin_data]
    combined_data = pd.concat(data_list, ignore_index=True)
    #combined_data = swin_data
    # print the average ICC of all combined data:
    for i in range(len(data_list)):
        print(f'Average ICC: {data_list[i]["ICC"].mean()}')
        print(f'Std ICC: {data_list[i]["ICC"].std()}')
    # Create and save the ICC boxplot
    #create_boxplot(combined_data, 'ICC', 'Distribution of ICC by Method', 'ICC', 'icc_boxplot.png')
    create_boxplot(combined_data, 'ICC', '', 'ICC', 'icc_boxplot.png')

    # Create and save the CCC boxplot
    #create_boxplot(combined_data, 'CCC', 'Distribution of CCC by Method', 'CCC', 'ccc_boxplot.png')

if __name__ == '__main__':
    main()
