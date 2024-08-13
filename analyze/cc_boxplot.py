import pandas as pd
import matplotlib.pyplot as plt

def load_data(file_path, method_name):
    """Load data from a CSV file and add a method column."""
    data = pd.read_csv(file_path)
    data['Method'] = method_name
    return data

def create_boxplot(data, value_column, title, y_label, output_path):
    """Create and save a boxplot for a specified value column."""
    fig, ax = plt.subplots(figsize=(7, 7))
    data.boxplot(column=value_column, by='Method', ax=ax)
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xlabel('Feature Extraction Method')
    plt.suptitle('')
    plt.tight_layout()
    plt.show()
    fig.savefig(output_path)

def main():
    # Load data for different methods
    swin_data = load_data('icc_features_swinunetr_full.csv', ' Swin')
    finetuneswin_data = load_data('icc_features_paper_contrastive2_F1.csv', ' rois Swin')
    randomcrop_swin_data = load_data('icc_features_random_contrast_5_6_lowLR_12batch_swin.csv', ' randomcrop Swin')
    livercrops_swin_data = load_data('nicc_features_liverrandom_contrast_5_15_10batch_swin.csv', ' livercrops Swin')
    combat_swin_data = load_data('icc_combat_features_swinunetr_full.csv', 'combat Swin')
    pyradiomics_data = load_data('icc_features_pyradiomics_full.csv', 'Radiomics')
    cnn_data = load_data('icc_features_oscar_full.csv', 'cnn')
    combat_cnn_data = load_data('icc_2combat_features_oscar_full.csv', 'combat cnn')
    combat_pyradiomics_data = load_data('icc_combat_features_pyradiomics_full.csv', ' combat Radiomics')
    

    # Combine data into a single DataFrame
    combined_data = pd.concat([swin_data, finetuneswin_data, randomcrop_swin_data, livercrops_swin_data], ignore_index=True)
    #combined_data = swin_data

    # Create and save the ICC boxplot
    create_boxplot(combined_data, 'ICC', 'Distribution of ICC by Method', 'ICC', 'icc_boxplot.png')

    # Create and save the CCC boxplot
    #create_boxplot(combined_data, 'CCC', 'Distribution of CCC by Method', 'CCC', 'ccc_boxplot.png')

if __name__ == '__main__':
    main()
