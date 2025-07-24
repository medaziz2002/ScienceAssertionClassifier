# balancer.py

import pandas as pd
import numpy as np
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
from collections import Counter
import matplotlib.pyplot as plt


# --- Class Distribution Functions ---

def check_class_distribution(y):
    """Check the class distribution of a target variable."""
    class_counts = Counter(y)
    print("Class distribution:")
    for class_label, count in class_counts.items():
        print(f"Class {class_label}: {count} samples ({count/len(y)*100:.2f}%)")
    return class_counts


def plot_class_distribution(class_counts, title="Class Distribution"):
    """Plot the class distribution."""
    plt.figure(figsize=(10, 6))
    plt.bar(class_counts.keys(), class_counts.values())
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Count")
    for i, count in enumerate(class_counts.values()):
        plt.text(i, count + 0.1, str(count), ha='center')
    plt.tight_layout()
    plt.savefig(f"{title.lower().replace(' ', '_')}.png")
    plt.close()


# --- Sampling Methods ---

def random_oversampling(X, y, random_state=42):
    """Apply random oversampling to the minority class."""
    ros = RandomOverSampler(random_state=random_state)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    return X_resampled, y_resampled


def random_undersampling(X, y, random_state=42):
    """Apply random undersampling to the majority class."""
    rus = RandomUnderSampler(random_state=random_state)
    X_resampled, y_resampled = rus.fit_resample(X, y)
    return X_resampled, y_resampled


def smote_oversampling(X, y, random_state=42):
    """Apply SMOTE oversampling to the minority class."""
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled


def smote_tomek_sampling(X, y, random_state=42):
    """Apply SMOTE-Tomek hybrid sampling."""
    smote_tomek = SMOTETomek(random_state=random_state)
    X_resampled, y_resampled = smote_tomek.fit_resample(X, y)
    return X_resampled, y_resampled


def smote_enn_sampling(X, y, random_state=42):
    """Apply SMOTE-ENN hybrid sampling."""
    smote_enn = SMOTEENN(random_state=random_state)
    X_resampled, y_resampled = smote_enn.fit_resample(X, y)
    return X_resampled, y_resampled


# --- Class Weights ---

def balanced_class_weights(y):
    """Calculate balanced class weights."""
    class_counts = Counter(y)
    n_samples = len(y)
    weights = {class_label: n_samples / (len(class_counts) * count)
               for class_label, count in class_counts.items()}
    return weights


# --- Generic Balancing Function ---

def apply_data_balancing(X, y, method='smote', random_state=42):
    """Apply a specified data balancing method."""
    print(f"Applying {method} balancing...")

    if method == 'random_oversampling':
        X_resampled, y_resampled = random_oversampling(X, y, random_state)
    elif method == 'random_undersampling':
        X_resampled, y_resampled = random_undersampling(X, y, random_state)
    elif method == 'smote':
        X_resampled, y_resampled = smote_oversampling(X, y, random_state)
    elif method == 'smote_tomek':
        X_resampled, y_resampled = smote_tomek_sampling(X, y, random_state)
    elif method == 'smote_enn':
        X_resampled, y_resampled = smote_enn_sampling(X, y, random_state)
    else:
        raise ValueError(f"Unknown balancing method: {method}")

    print(f"Original dataset shape: {Counter(y)}")
    print(f"Resampled dataset shape: {Counter(y_resampled)}")

    return X_resampled, y_resampled


# --- Visualization of Comparisons ---

def compare_distributions(distributions, title="Comparison of Class Distributions"):
    """Compare multiple class distributions."""
    plt.figure(figsize=(12, 8))

    methods = list(distributions.keys())
    classes = sorted(list(next(iter(distributions.values())).keys()))

    x = np.arange(len(methods))
    width = 0.8 / len(classes)

    for i, class_label in enumerate(classes):
        counts = [distributions[method].get(class_label, 0) for method in methods]
        plt.bar(x + i * width, counts, width, label=f'Class {class_label}')

    plt.xlabel('Method')
    plt.ylabel('Count')
    plt.title(title)
    plt.xticks(x + width * (len(classes) - 1) / 2, methods, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


# --- Test Section ---

if __name__ == "__main__":
    # Example test using synthetic data

    data = pd.read_csv('scitweets_export.tsv', sep='\t')
    print("Dataset loaded successfully!")
    print(f"Dataset shape: {data.shape}")
    print("\nFirst few rows:")
    print(data.head())

    # Check for missing values
    print("\nMissing values:")
    print(data.isnull().sum())

    print("\nPreparing the data for classification...")

    numeric_cols = data.select_dtypes(include=['number']).columns.tolist()

    if len(numeric_cols) >= 5:
        feature_cols = numeric_cols[:5]
        target_col = data.columns[-1]
    else:
        print("Not enough numeric features found. Creating synthetic features for demonstration.")
        for i in range(5):
            data[f'synthetic_feature_{i}'] = np.random.randn(len(data))
        feature_cols = [f'synthetic_feature_{i}' for i in range(5)]

        if 'target' not in data.columns:
            data['target'] = np.random.choice([0, 1], size=len(data), p=[0.8, 0.2])
        target_col = 'target'

    X = data[feature_cols].values
    y = data[target_col].values

    if np.isnan(X).any() or np.isnan(y).any():
        print("Found NaN values in data. Filling with zeros...")
        X = np.nan_to_num(X)
        y = np.nan_to_num(y)

    df = pd.DataFrame(X, columns=feature_cols)
    df['target'] = y

    print("\nProcessed DataFrame:")
    print(df.head())
    print(f"Shape: {df.shape}")

    original_class_counts = check_class_distribution(y)
    plot_class_distribution(original_class_counts, "Original Class Distribution")

    print("\nBalanced class weights:")
    weights = balanced_class_weights(y)
    print(weights)

    print("\nTesting various resampling methods:")

    X_ros, y_ros = random_oversampling(X, y)
    ros_class_counts = check_class_distribution(y_ros)

    X_rus, y_rus = random_undersampling(X, y)
    rus_class_counts = check_class_distribution(y_rus)

    X_smote, y_smote = smote_oversampling(X, y)
    smote_class_counts = check_class_distribution(y_smote)

    X_smote_tomek, y_smote_tomek = smote_tomek_sampling(X, y)
    smote_tomek_class_counts = check_class_distribution(y_smote_tomek)

    X_smote_enn, y_smote_enn = smote_enn_sampling(X, y)
    smote_enn_class_counts = check_class_distribution(y_smote_enn)

    distributions = {
        'Original': original_class_counts,
        'Random Over': ros_class_counts,
        'Random Under': rus_class_counts,
        'SMOTE': smote_class_counts,
        'SMOTE-Tomek': smote_tomek_class_counts,
        'SMOTE-ENN': smote_enn_class_counts
    }

    compare_distributions(distributions, "Comparison of Resampling Methods")

    print("\nTesting generic balancing function with SMOTE:")
    X_balanced, y_balanced = apply_data_balancing(X, y, method='smote')

    df_resampled = pd.DataFrame(X_balanced, columns=feature_cols)
    df_resampled['target'] = y_balanced

    print("\nResampled DataFrame (using SMOTE):")
    print(df_resampled.head())
    print(f"Shape: {df_resampled.shape}")

    print("\nSummary of resampling methods:")
    for method, counts in distributions.items():
        class_counts = ", ".join([f"Class {c}: {n}" for c, n in counts.items()])
        print(f"{method}: {class_counts}")
