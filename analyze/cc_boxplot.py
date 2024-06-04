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
    fig.savefig(output_path)

def main():
    # Load data for different methods
    swin_data = load_data('../../all_dataset_features/icc_features_swinunetr_full.csv', 'Swin')
    pyradiomics_data = load_data('../../all_dataset_features/pca_pyradiomics_icc_results5.csv', 'Pyradiomics')
    cnn_data = load_data('../../all_dataset_features/features_ocar_full_icc_results.csv', 'Oscar cnn')

    # Combine data into a single DataFrame
    combined_data = pd.concat([swin_data, pyradiomics_data,cnn_data], ignore_index=True)
    #combined_data = swin_data

    # Create and save the ICC boxplot
    create_boxplot(combined_data, 'ICC', 'Distribution of ICC by Method', 'ICC', 'icc_boxplot.png')

    # Create and save the CCC boxplot
    #create_boxplot(combined_data, 'CCC', 'Distribution of CCC by Method', 'CCC', 'ccc_boxplot.png')

if __name__ == '__main__':
    main()
