#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Hardcoded parameters
input_path = "/home/preben/Documents/Master/merged_hvac_dataset.csv"
output_dir = "/home/preben/Pictures/Master/Model Results"
n_components = 5  # Number of principal components to compute
# Specify a target column to compare PCs against (e.g., a column you want to predict)
target_column = "Preben - Meeting Room 434 Temperature"  # <-- set this to your column name

# Main execution
def main():
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "explained_variance.png")

    # Load data
    df = pd.read_csv(input_path, low_memory=False)

    # Check target column
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")
    y = df[target_column].ffill().bfill()
    if y.isnull().any():
        raise ValueError("Missing values remain in target column after fill.")

    # Select numeric features and exclude target
    df_numeric = df.select_dtypes(include=[np.number]).copy()
    if target_column in df_numeric.columns:
        df_numeric.drop(columns=[target_column], inplace=True)
        print(f"Excluded target column from PCA features: {target_column}")

    dropped = df.columns.difference(df_numeric.columns.tolist() + [target_column]).tolist()
    if dropped:
        print(f"Dropped non-numeric columns: {dropped}")

    # Handle missing values in features
    df_numeric.ffill(inplace=True)
    df_numeric.bfill(inplace=True)
    if df_numeric.isnull().any().any():
        raise ValueError("Missing values remain in features after fill.")

    # Standardize features
    X_scaled = StandardScaler().fit_transform(df_numeric.values)

    # Run PCA
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X_scaled)

    # Compute explained variance
    var_ratio = pca.explained_variance_ratio_
    cum_var = np.cumsum(var_ratio)

    # Determine component descriptions
    feature_names = df_numeric.columns.tolist()
    component_info = []
    for i, comp in enumerate(pca.components_):
        label = f"PC{i+1} - {feature_names[np.argmax(np.abs(comp))]}"
        corr = np.corrcoef(scores[:, i], y.values)[0, 1]
        top_idxs = np.argsort(np.abs(comp))[::-1][:5]
        top_feats = [(feature_names[j], comp[j]) for j in top_idxs]
        component_info.append({
            'label': label,
            'explained': var_ratio[i],
            'cumulative': cum_var[i],
            'corr': corr,
            'top_feats': top_feats
        })

    # Plot explained variance
    labels = [info['label'] for info in component_info]
    explained = [info['explained'] for info in component_info]
    cumulative = [info['cumulative'] for info in component_info]
    y_max = max(max(explained), max(cumulative)) * 1.2

    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels, explained, alpha=0.7)
    plt.plot(labels, cumulative, marker='o', linestyle='--', label='Cumulative')
    for bar, ev in zip(bars, explained):
        plt.text(bar.get_x() + bar.get_width()/2, ev + 0.01,
                 f"{ev*100:.1f}%", ha='center', va='bottom', fontsize=9)
    for i, cu in enumerate(cumulative):
        plt.text(i, cu + 0.01, f"{cu*100:.1f}%", ha='center', va='bottom', fontsize=9)

    plt.xlabel('Principal Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title('PCA Explained Variance with Component Descriptions')
    plt.ylim(0, y_max)
    plt.xticks(rotation=45, ha='right')
    plt.legend(loc='best')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_file)
    print(f"Explained variance plot saved to: {output_file}\n")

    # Nicely formatted terminal output
    print("PCA Component Summary:\n")
    for info in component_info:
        print(f"{info['label']}")
        print(f"  Explained Variance: {info['explained']*100:.2f}%")
        print(f"  Cumulative Variance: {info['cumulative']*100:.2f}%")
        print(f"  Corr with '{target_column}': {info['corr']:.3f}")
        print("  Top 5 Features:")
        for feat, loading in info['top_feats']:
            print(f"    - {feat}: {loading:.3f}")
        print()

if __name__ == "__main__":
    main()
