import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load data
def load_data(file_path):
    return pd.read_csv(file_path)

# Function to analyze negative ICC values
def analyze_negative_values(file_path):

    data = load_data(file_path)
    
    # Find negative values in the 'ICC' column
    negative_values = data[data['ICC'] < 0]
    num_negative_values = len(negative_values)
    
    # If there are negative values, extract the features and their values
    if num_negative_values > 0:
        most_negative_icc = data['ICC'].min() if num_negative_values > 0 else None
        features_with_negative_values = negative_values[['Feature', 'ICC']]
    else:
        most_negative_icc = None
        features_with_negative_values = None
    
    return num_negative_values, features_with_negative_values, most_negative_icc

# Function to generate a bar plot of number of negative ICCs per scanner
def plot_negative_iccs_bar(methods, scanner, negative_values_dict, method_labels, ax):
    method_names = [method_labels.get(method, method) for method in methods]
    num_negative_values = [negative_values_dict[method] for method in methods]

    colors = ['#FF6347', '#32CD32', '#1E90FF']
    bars = ax.bar(method_names, num_negative_values, color=colors)
    ax.set_title(f"Negative ICC Values for Scanner {scanner}")
    ax.set_xlabel('Method')
    ax.set_ylabel('Number of Negative ICCs')
    ax.set_ylim(0, 325)
    
    # Add the number of negative values on top of each bar
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 1, str(int(yval)), ha='center', color='black', fontsize=12)

# Function to generate the report
def generate_report(files_dir, methods, scanners):
    method_labels = {
        'pyradiomics': 'Pyradiomics',
        'cnn': 'Shallow CNN',
        'swinunetr': 'SwinUNETR'
    }

    report = []
    negative_values_per_scanner = {scanner: {} for scanner in scanners}

    # Create a 2x2 subplot grid for each scanner
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    for scanner_idx, scanner in enumerate(scanners):
        ax = axes[scanner_idx // 2, scanner_idx % 2]  # Get the appropriate subplot axis

        for method in methods:
            file_path = f'{files_dir}/icc_dose_features_{method}_full_{scanner}.csv'
            
            num_negative_values, features_with_negative_values, min_icc = analyze_negative_values(file_path)
            report.append(f"Method: {method_labels.get(method, method)}, Scanner: {scanner}\n")
            report.append(f"Number of negative ICC values: {num_negative_values}"+"\n")
            
            if num_negative_values > 0:
                report.append(f"Most negative ICC value: {min_icc}\n")
                report.append("Features with negative ICC values:\n")
                report.append(features_with_negative_values.to_string(index=False)+"\n")
            else:
                report.append("No negative ICC values found.")
            
            report.append("\n" + "-"*50 + "\n")
            
            negative_values_per_scanner[scanner][method] = num_negative_values

        # Generate the bar plot for the number of negative ICCs for each method in the scanner
        plot_negative_iccs_bar(methods, scanner, negative_values_per_scanner[scanner], method_labels, ax)

    plt.tight_layout()
    plt.savefig(f"/mnt/nas7/data/maria/final_features/icc_results_dose/icc_report/iccs_negative_values.png")
    plt.close()

    # Save the report as a text file
    output_dir = '/mnt/nas7/data/maria/final_features/icc_results_dose/icc_report'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'icc_negative_values_report.txt')

    with open(output_path, 'w') as f:
        f.writelines(report)

    print(f"Report saved at {output_path}")

# Main function
def main():
    files_dir = '/mnt/nas7/data/maria/final_features/icc_results_dose'
    methods = ['pyradiomics', 'cnn', 'swinunetr']
    scanners = ['A1', 'A2', 'B1', 'B2']

    generate_report(files_dir, methods, scanners)

if __name__ == "__main__":
    main()


