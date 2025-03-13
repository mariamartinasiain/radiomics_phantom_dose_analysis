import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Function to load data
def load_data(file_path):
    return pd.read_csv(file_path)

# Function to compute statistics for each method and scanner
def compute_statistics(data, methods):
    icc_data = {}

    for method in methods:
        if method in data:
            icc_values = data[method]['ICC'].dropna()
            icc_data[method] = icc_values 

    # Compute statistics for each method
    stats_per_method = {}
    for method, values in icc_data.items():
        stats_per_method[method] = {
            'Min': np.min(values),
            'Max': np.max(values),
            'Mean': np.mean(values),
            'Median': np.median(values),
            'Std': np.std(values),
            'IQR': np.percentile(values, 75) - np.percentile(values, 25),
            'N': len(values)
        }

    return stats_per_method

# Function to create a DataFrame with statistics for each scanner and method
def create_statistics_dataframe(statistics, scanners, methods):
    rows = []
    for scanner in scanners:

        for method in methods:
            if method in statistics[scanner]:
                stats = statistics[scanner][method]
                rows.append({
                    'Scanner': scanner,
                    'Method': method,
                    'Mean ± Std': f"{stats['Mean']:.4f} ± {stats['Std']:.4f}",
                    'Median ± Std': f"{stats['Median']:.4f} ± {stats['Std']:.4f}"
                })
    
    df = pd.DataFrame(rows)
    return df

# Function to save statistics to a CSV file
def save_statistics_to_csv(df, output_dir):
    csv_file_path = os.path.join(output_dir, 'icc_statistics_pca.csv')
    df.to_csv(csv_file_path, index=False)
    print(f"CSV file with statistics saved to {csv_file_path}")

# Function to save statistics to a single text file for all scanners
def save_statistics_all_scanners(statistics, methods, scanners, output_dir):
    stats_file_path = os.path.join(output_dir, 'icc_statistics_all_scanners_pca.txt')
    
    with open(stats_file_path, 'w') as file:
        for scanner in scanners:
            file.write(f"Scanner: {scanner}\n")
            for method in methods:
                if method in statistics[scanner]:
                    file.write(f"  Method: {method}\n")
                    for stat_name, value in statistics[scanner][method].items():
                        file.write(f"    {stat_name}: {value}\n")
            file.write("\n")  
    print(f"All statistics saved to {stats_file_path}")

# Function to accumulate statistics for all scanners
def accumulate_statistics(all_data, methods, scanners):
    statistics_all_scanners = {}


    for scanner in scanners:
        statistics_all_scanners[scanner] = compute_statistics(all_data[scanner], methods)

    return statistics_all_scanners

# Function to create boxplots for each scanner and method
def create_boxplots(ax, data, methods, title):
    data_to_plot = [data[method]['ICC'].dropna() for method in methods if method in data] #Removed the line that excludes ICC values that are 0

    custom_labels = ['Pyradiomics', 'Shallow CNN', 'SwinUNETR', 'CT-FM']

    box = ax.boxplot(data_to_plot, labels=custom_labels, showfliers=False)

    for i, method in enumerate(data.keys()):
        icc_filtered = data[method]['ICC'].dropna() #Removed the line that excludes ICC values that are 0
        n = len(icc_filtered)
        ax.text(i + 1, 1.02, f'N={n}', ha='center', va='bottom', fontsize=12)

    ax.set_title(f'{title}', fontsize=14, fontweight='bold', pad=28)
    ax.set_ylabel('ICC', fontsize=12)
    #ax.set_ylim(0.1, 1)
    ax.grid(True, axis='y', linestyle="solid", linewidth=0.5)
    #ax.set_yticks(np.arange(0.1, 1.05, 0.05))
    ax.tick_params(axis='x', labelsize=14)


# Function to create a merged boxplot for all scanners
def save_median_std_merged(data_to_plot, methods, output_dir):
    stats_file_path = os.path.join(output_dir, 'merged_icc_median_std_pca.txt')
    with open(stats_file_path, 'w') as file:
        for method in methods:
            values = data_to_plot[method]
            if values:
                median = np.median(values)
                std = np.std(values)
                mean = np.mean(values)
                file.write(f"{method}\t{median:.4f}\t{mean:.4f} ± {std:.4f}\n")

def create_merged_boxplot(all_data, methods, scanners, output_dir):
    data_to_plot = {method: [] for method in methods}
    for scanner in scanners:
        for method in methods:
            if method in all_data[scanner]:
                icc_values = all_data[scanner][method]['ICC'].dropna()
                data_to_plot[method].extend(icc_values)
    plot_data = [data_to_plot[method] for method in methods]
    fig, ax = plt.subplots(figsize=(8, 6))
    box = ax.boxplot(plot_data, labels=['Pyradiomics', 'Shallow CNN', 'SwinUNETR', 'CT-FM'], showfliers=False)
    for i, method in enumerate(methods):
        n = len(data_to_plot[method])
        max_val = max(data_to_plot[method]) if len(data_to_plot[method]) > 0 else 0.9
        ax.text(i + 1, max_val + 0.02, f'N={n}', ha='center', va='bottom', fontsize=12)
    ax.set_title("ICC Comparison Across Methods (All 13 Scanners Merged)", fontsize=14, fontweight='bold', pad=25)
    ax.set_ylabel("ICC", fontsize=14)
    ax.grid(True, axis='y', linestyle="solid", linewidth=0.5)
    #ax.set_yticks(np.arange(0.65, 1.025, 0.025))
    ax.tick_params(axis='x', labelsize=13)
    output_path = os.path.join(output_dir, "merged_icc_boxplot_pca.png")
    plt.savefig(output_path, dpi=300)
    plt.show()
    print(f'Merged boxplot saved to {output_path}')
    save_median_std_merged(data_to_plot, methods, output_dir)


def main():
    files_dir = '/mnt/nas7/data/maria/final_features/icc_results_dose/four_rois/pca'
    #files_dir = '/mnt/nas7/data/maria/final_features/icc_results_dose/six_rois'
    methods = ['pyradiomics', 'cnn', 'swinunetr', 'ct-fm']
    scanners = ["A1", "A2", "B1", "B2", "G1", "G2", "C1", "H2", "D1", "E2", "F1", "E1", "H1"]

    #output_dir = '/mnt/nas7/data/maria/final_features/icc_results_dose/icc_boxplot/six_rois'
    output_dir = '/mnt/nas7/data/maria/final_features/icc_results_dose/icc_boxplot/four_rois/pca'
    os.makedirs(output_dir, exist_ok=True)

    #statistics_output_dir = '/mnt/nas7/data/maria/final_features/icc_results_dose/six_rois'
    statistics_output_dir = '/mnt/nas7/data/maria/final_features/icc_results_dose/four_rois/pca'
    os.makedirs(statistics_output_dir, exist_ok=True)

    all_data = {scanner: {} for scanner in scanners}

    for method in methods:
        for scanner in scanners:
            #file_path = f'{files_dir}/icc_scanner_{scanner}/icc_dose_features_{method}_full_{scanner}.csv'
            file_path = f'{files_dir}/icc_scanner_{scanner}/icc_dose_features_{method}_full_{scanner}_pca.csv'
            icc_data = load_data(file_path)
            icc_data = icc_data[icc_data['Feature'] != 'ROI_numerical']
            all_data[scanner][method] = icc_data

    for scanner in scanners:
        fig, ax = plt.subplots(figsize=(12, 8))
        create_boxplots(ax, all_data[scanner], methods, f'Scanner {scanner}')
        output_path = os.path.join(output_dir, f'{scanner}_boxplot_dose_comparison_pca.png')
        fig.savefig(output_path, dpi=300)
        print(f'Plot for {scanner} saved.')

    statistics_all_scanners = accumulate_statistics(all_data, methods, scanners)
    save_statistics_all_scanners(statistics_all_scanners, methods, scanners, statistics_output_dir)
    stats_df = create_statistics_dataframe(statistics_all_scanners, scanners, methods)
    save_statistics_to_csv(stats_df, statistics_output_dir)
    create_merged_boxplot(all_data, methods, scanners, output_dir)


if __name__ == "__main__":
    main()
