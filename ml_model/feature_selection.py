"""
feature_selection.py
─────────────────────
Uses Random Forest to rank feature importance and select
the top N most relevant features for DDoS detection.

This is the FIRST stage of the hybrid SVM-RF pipeline:
  RF → selects features → SVM trains on selected features

Usage:
    from feature_selection import select_features
    X_reduced, indices = select_features(X, y)
"""

import numpy as np
import matplotlib.pyplot as plt
import joblib, os
from sklearn.ensemble import RandomForestClassifier

RESULTS_DIR = '../results/plots/'
os.makedirs(RESULTS_DIR, exist_ok=True)


def select_features(X, y, top_n=15, save_plot=True):
    """
    Trains a Random Forest and selects the top_n most important features.

    Parameters:
        X       : numpy array of shape (n_samples, n_features)
        y       : numpy array of labels
        top_n   : number of features to keep (default 15)
        save_plot: whether to save a feature importance bar chart

    Returns:
        X_reduced : numpy array with only top_n features
        indices   : indices of selected features (for use at inference)
    """
    print(f"\n── RF Feature Selection (top {top_n} features) ──")
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1        # use all CPU cores
    )
    rf.fit(X, y)

    importances = rf.feature_importances_
    indices     = np.argsort(importances)[::-1][:top_n]

    print(f"  Total features: {X.shape[1]}")
    print(f"  Selected features (indices): {indices.tolist()}")
    print(f"  Top feature importances:")
    for rank, idx in enumerate(indices):
        print(f"    #{rank+1:2d}  Feature[{idx:3d}]  importance={importances[idx]:.4f}")

    # Save selected indices for inference
    joblib.dump(indices, '../ml_model/saved_model/feature_indices.pkl')
    print("  Feature indices saved → ml_model/saved_model/feature_indices.pkl")

    # Plot feature importances
    if save_plot:
        plt.figure(figsize=(12, 5))
        plt.bar(range(top_n), importances[indices], color='steelblue')
        plt.title(f'Top {top_n} Feature Importances (Random Forest)')
        plt.xlabel('Feature Rank')
        plt.ylabel('Importance Score')
        plt.tight_layout()
        path = os.path.join(RESULTS_DIR, 'feature_importance.png')
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Feature importance plot saved → {path}")

    X_reduced = X[:, indices]
    print(f"  Reduced feature matrix shape: {X_reduced.shape}")
    return X_reduced, indices


if __name__ == '__main__':
    # Quick test with random data
    print("Running feature selection on dummy data (for testing)...")
    X_dummy = np.random.rand(500, 30)
    y_dummy = np.random.randint(0, 2, 500)
    X_reduced, idx = select_features(X_dummy, y_dummy, top_n=10)
    print(f"Output shape: {X_reduced.shape}")
