import pandas as pd
import matplotlib.pyplot as plt

# Correspondance des indices avec les noms des scanners
scanner_names = {0: 'A1', 1: 'A2', 2: 'B1', 3: 'B2', 4: 'C1', 5: 'D1', 6: 'E1', 7: 'E2', 8: 'F1', 9: 'G1', 10: 'G2', 11: 'H1', 12: 'H2'}

def load_and_process_data(file_path):
    data = pd.read_csv(file_path, header=None)
    accuracy_data = data.iloc[1:, :-2].astype(float)
    lowest_accuracy_row = accuracy_data.mean(axis=1).idxmin()
    lowest_accuracy_values = accuracy_data.loc[lowest_accuracy_row]
    filtered_accuracy_data = accuracy_data.drop(lowest_accuracy_row)
    filtered_mean_accuracies = filtered_accuracy_data.mean()

    # Calculate variance for each concentration across all splits
    variances = accuracy_data.var()
    mean_variance = variances.mean()

    return filtered_mean_accuracies, lowest_accuracy_values, lowest_accuracy_row, mean_variance, accuracy_data

def plot_comparison(mean_accuracies_dict, lowest_accuracies_dict, lowest_accuracy_rows, variances_dict, all_accuracies_dict):
    plt.figure(figsize=(12, 8))
    max_len = max(len(accuracies) for accuracies in mean_accuracies_dict.values())
    x_values = list(range(1, max_len + 1))
    
    for label, accuracies in mean_accuracies_dict.items():
        plt.plot(x_values[:len(accuracies)], accuracies, marker='o', linestyle='-', label=f'{label} Mean Accuracy')
        variance = variances_dict[label]
        plt.text(len(accuracies), accuracies.iloc[-1], f'Var: {variance:.4f}', fontsize=9, verticalalignment='center')
    plt.title('Filtered Mean Accuracy vs. Number of Scanners Used for Training')
    plt.xlabel('Number of Scanners Used for Training')
    plt.ylabel('Filtered Mean Accuracy')
    plt.xticks(x_values)
    plt.grid(True)
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(12, 12))
    for label, accuracies in lowest_accuracies_dict.items():
        scanner_name = 'H2y'
        plt.plot(x_values[:len(accuracies)], accuracies, marker='o', linestyle='-', label=f'{label} Lowest Accuracy Run ({scanner_name})')
        # Annotate variance
        variance = variances_dict[label]
        plt.text(len(accuracies), accuracies.iloc[-1], f'Var: {variance:.4f}', fontsize=9, verticalalignment='center')
    plt.title('Lowest Accuracy Run vs. Number of Scanners Used for Training')
    plt.xlabel('Number of Scanners Used for Training')
    plt.ylabel('Lowest Accuracy')
    plt.xticks(x_values)
    plt.grid(True)
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(12, 8))
    for label, accuracies in all_accuracies_dict.items():
        mean_accuracies = accuracies.mean(axis=0)
        plt.plot(x_values[:len(mean_accuracies)], mean_accuracies, marker='o', linestyle='-', label=f'{label} All Rows Mean Accuracy')
        variance = variances_dict[label]
        plt.text(len(mean_accuracies), mean_accuracies.iloc[-1], f'Var: {variance:.4f}', fontsize=9, verticalalignment='center')
    plt.title('Mean Accuracy of All Rows vs. Number of Scanners Used for Training')
    plt.xlabel('Number of Scanners Used for Training')
    plt.ylabel('Mean Accuracy of All Rows')
    plt.xticks(x_values)
    plt.grid(True)
    plt.legend()
    plt.show()

def main():
    files = {
        #'SwinUNETR': '../../accu/results_roi_large__None_features_swinunetr_full.csv.csv',
        # 'OSCAR': '../../accu/results_roi_large__None_features_oscar_full.csv.csv',
        # 'Pyradiomics': '../../accu/results_roi_large__None_features_pyradiomics_full.csv.csv',
        # 'SwinFinetune': '../../accu/results_roi_large__None_features_swin_finetune.csv.csv',
        # 'SwinUNETRAveraged': '../../accu/results_roi_large__None_features_swin_finetune_averaged.csv.csv',
        # 'SwinFinetuneAveraged': '../../accu/results_roi_large__None_features_swinunetr_full_averaged.csv.csv',
        
#         results_roi_large_999_None_features_combat_oscar_full.csv                                                                                             100% 1232    37.0KB/s   00:00    
# results_roi_large_999_None_features_combat_pyradiomics_full.csv                                                                                       100% 1842    56.8KB/s   00:00    
# results_roi_large_999_None_features_combat_swinunetr_full.csv   
# results_roi_large_999_None_features_contrast_oscar.csv 
        # 'Oscar combat': 'results_scanner_999_None_features_combat_oscar_full.csv',
        # 'Oscar contrast': 'results_scanner_999_None_features_contrast_oscar.csv',
        # 'Oscar' : 'results_scanner__None_features_oscar_full.csv.csv',
        # 'Pyradiomics combat': 'results_scanner_999_None_features_combat_pyradiomics_full.csv',
        # 'Pyradiomics': 'results_scanner__None_features_pyradiomics_full.csv.csv',
        # 'SwinUNETR combat': 'results_scanner_999_None_features_combat_swinunetr_full.csv',
        # 'SwinUNETR random crop contrastive': 'results_scanner_999_None_features_random_contrast_5_6_lowLR_12batch_swin.csv',
        # 'SwinFinetune' : 'results_scanner__None_features_paper_contrastive2_F1.csv',
        # 'SwinUNETR': 'results_scanner__None_features_swinunetr_full.csv.csv',
        # 'swin liver' : 'results_roi_small_999_None_features_liverrandom_contrast_5_15_10batch_swin.csv',
        'Oscar combat': 'results_roi_large_999_None_features_combat_oscar_full.csv',
        # # 'Oscar contrast': 'results_roi_large_999_None_features_contrast_oscar.csv',
        # 'Oscar' : 'results_roi_large__None_features_oscar_full.csv.csv',
        'Pyradiomics combat': 'results_roi_large_999_None_features_combat_pyradiomics_full.csv',
        # 'Pyradiomics': 'results_roi_large__None_features_pyradiomics_full.csv.csv',
        'SwinUNETR combat': 'results_roi_large_999_None_features_combat_swinunetr_full.csv',
        # 'SwinUNETR random crop contrastive' : 'results_roi_small_999_None_features_random_contrast_registered.csv',
        # 'SwinUNETR random contrast 8 8' : 'results_roi_small_999_None_features_random_contrast_8_8_swin.csv',
        # 'SwinUNETR random crop contrastive 5 6 low LR': 'results_roi_small_999_None_features_random_contrast_5_6_lowLR_12batch_swin.csv',
        # 'SwinFinetune' : 'results_roi_large_crossval_trained_None_swinunetr_paper.csv',
        # 'SwinUNETR': 'results_roi_large__None_features_swinunetr_full.csv.csv',
        # 'icc combat radiomics' : 'results_scanner_999_None_features_icc_cb_pyradiomics.csv',
        # 'icc combat oscar' : 'results_scanner_999_None_features_icc_cb_oscar.csv',
        # 'icc combat swin' : 'results_scanner_999_None_features_icc_cb_swinunetr.csv',
        'icc combat radiomics' : 'results_roi_large_999_None_features_icc_cb_pyradiomics.csv',
        'icc combat oscar' : 'results_roi_large_999_None_features_icc_cb_oscar.csv',
        'icc combat swin' : 'results_roi_large_999_None_features_icc_cb_swinunetr.csv',
    }

    mean_accuracies_dict = {}
    lowest_accuracies_dict = {}
    lowest_accuracy_rows = {}
    variances_dict = {}
    all_accuracies_dict = {}
    
    for label, file_path in files.items():
        mean_accuracies, lowest_accuracy_values, lowest_accuracy_row, mean_variance, all_accuracies = load_and_process_data(file_path)
        mean_accuracies_dict[label] = mean_accuracies
        lowest_accuracies_dict[label] = lowest_accuracy_values
        lowest_accuracy_rows[label] = lowest_accuracy_row
        variances_dict[label] = mean_variance
        all_accuracies_dict[label] = all_accuracies
        #printing all accuracies size and mean accuracies size
        print(f"Mean Accuracies size for {label}: {mean_accuracies.shape}")
        print(f"All Accuracies size for {label}: {all_accuracies.shape}")
        

    plot_comparison(mean_accuracies_dict, lowest_accuracies_dict, lowest_accuracy_rows, variances_dict, all_accuracies_dict)

if __name__ == "__main__":
    main()
