import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from scipy.stats import skew, kurtosis, ks_2samp


# Output directory
output_dir = "/mnt/nas7/data/maria/final_features/dose_analysis"
os.makedirs(output_dir, exist_ok=True)

# Helper functions
def extract_mg_value(series_description):
    match = re.search(r'(\d+)mGy', series_description)
    return int(match.group(1)) if match else None

def load_features(path, method_name):
    df = pd.read_csv(path)
    df['Dose'] = df['SeriesDescription'].apply(extract_mg_value)
    df = df[df['Dose'].isin([1, 14])]  # Keep only 1mGy and 14mGy

    # Assume features are already in columns, except for deep features (for CT-FM)
    if 'deepfeatures' in df.columns:
        df['deepfeatures'] = df['deepfeatures'].apply(lambda x: np.fromstring(x.strip("[]"), sep=','))
        feature_df = pd.DataFrame(df['deepfeatures'].tolist(), index=df.index)
    else:
        non_feature_cols = ['StudyInstanceUID', 'SeriesNumber', 'SeriesDescription', 'ROI', 'ManufacturerModelName',
                            'Manufacturer', 'SliceThickness', 'SpacingBetweenSlices', 'FileName',
                            'StudyID', 'StudyDescription', 'Scanner', 'Dose']
        feature_df = df.drop(columns=[c for c in non_feature_cols if c in df.columns])

    feature_df['Dose'] = df['Dose'].values
    feature_df['ROI'] = df['ROI'].values
    feature_df['Method'] = method_name
    return feature_df

# Load radiomics data
pyrad = load_features("/mnt/nas7/data/maria/final_features/final_features_complete/features_pyradiomics_4rois.csv", "PyRadiomics")
pyrad.to_csv(os.path.join(output_dir, "pyradiomics_features_dose_filtered.csv"), index=False)

# Plot distributions in subplots
def plot_distributions(df, method, output_dir, max_features=24):
    features = df.drop(columns=["Dose", "Method"]).columns[:max_features]  # first N features
    n_features = len(features)
    
    # Create subplots
    fig, axes = plt.subplots(6, 8, figsize=(26, 18)) 
    axes = axes.flatten()
    
    for i, feat in enumerate(features):
        sns.kdeplot(data=df[df["Dose"] == 1], x=feat, label="1 mGy", fill=True, alpha=0.5, ax=axes[i])
        sns.kdeplot(data=df[df["Dose"] == 14], x=feat, label="14 mGy", fill=True, alpha=0.5, ax=axes[i])
        axes[i].set_title(f"{feat}")
        axes[i].set_xlabel(feat)
        axes[i].set_ylabel("Density")
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{method}_dose_comparison_subplot_48.png"))
    plt.close()

def plot_feature_statistics(df, method, output_dir, max_features=24):
    features = df.drop(columns=["Dose", "Method"]).columns[:max_features]
    
    stats_dict = {
        "Feature": [],
        "Metric": [],
        "Dose": [],
        "Value": []
    }

    ks_pvalues = []

    for feat in features:
        data_1 = df[df["Dose"] == 1][feat]
        data_14 = df[df["Dose"] == 14][feat]

        # Compute metrics
        stats = {
            "Mean": (np.mean(data_1), np.mean(data_14)),
            "Median": (np.median(data_1), np.median(data_14)),
            "Variance": (np.var(data_1), np.var(data_14)),
            "Skewness": (skew(data_1), skew(data_14)),
            "Kurtosis": (kurtosis(data_1), kurtosis(data_14)),
        }

        for metric, (val1, val14) in stats.items():
            stats_dict["Feature"].extend([feat, feat])
            stats_dict["Metric"].extend([metric, metric])
            stats_dict["Dose"].extend(["1 mGy", "14 mGy"])
            stats_dict["Value"].extend([val1, val14])

        # KS Test
        ks_stat, ks_pval = ks_2samp(data_1, data_14)
        ks_pvalues.append((feat, ks_pval))

    stats_df = pd.DataFrame(stats_dict)

    # Plot 1: general metrics
    g = sns.catplot(data=stats_df, x="Feature", y="Value", hue="Dose", col="Metric",
                    kind="bar", col_wrap=2, height=4, aspect=1.5, sharey=False)
    g.set_xticklabels(rotation=90)
    g.set_titles("{col_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{method}_feature_statistics_bars.png"))
    plt.close()

    # Plot 2: p-values KS test
    ks_df = pd.DataFrame(ks_pvalues, columns=["Feature", "KS_pvalue"])
    plt.figure(figsize=(12, 6))
    sns.barplot(data=ks_df, x="Feature", y="KS_pvalue", color="gray")
    plt.axhline(0.05, color="red", linestyle="--", label="p = 0.05")
    plt.xticks(rotation=90)
    plt.title(f"Kolmogorov-Smirnov p-values ({method})")
    plt.ylabel("p-value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{method}_ks_pvalues.png"))
    plt.close()


#plot_feature_statistics(pyrad, "CT-FM", output_dir, max_features=24)
# Plot for PyRadiomics method, limiting to first 20 features
#plot_distributions(pyrad, "CT-FM", output_dir, max_features=48)


def plot_distributions(df, method, output_dir, max_features=24):
    features = df.drop(columns=["Dose", "Method", "ROI"]).columns[:max_features]  # Exclude non-feature columns
    n_features = len(features)

    fig, axes = plt.subplots(4, 6, figsize=(26, 18)) 
    axes = axes.flatten()

    for i, feat in enumerate(features):
        # 1 mGy: solid lines
        sns.kdeplot(
            data=df[df["Dose"] == 1], 
            x=feat, 
            hue="ROI", 
            fill=True, 
            alpha=0.3, 
            ax=axes[i], 
            linewidth=1.5,
            common_norm=False,
            legend=True,
        )
        
        # 14 mGy: dashed lines
        sns.kdeplot(
            data=df[df["Dose"] == 14], 
            x=feat, 
            hue="ROI", 
            fill=False, 
            alpha=0.7, 
            ax=axes[i], 
            linestyle="--", 
            linewidth=1.5,
            common_norm=False,
            legend=True,
        )

        axes[i].set_title(f"{feat}")
        axes[i].set_xlabel(feat)
        axes[i].set_ylabel("Density")

    # Add a global legend outside the subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="ROI", loc="upper center", ncol=4)

    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Make space for the legend
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{method}_dose_roi_comparison_subplot.png"))
    plt.close()

plot_distributions(pyrad, "Pyradiomics", output_dir, max_features=24)
