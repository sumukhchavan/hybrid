"""
evaluate.py
───────────
Loads saved models, runs predictions, and generates all result plots.

Outputs (saved to results/plots/):
  - confusion_matrix.png
  - roc_curve.png
  - accuracy_comparison.png

Usage:
    python3 evaluate.py
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, os
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, f1_score,
    roc_curve, auc
)
from sklearn.model_selection import train_test_split

PROCESSED_DIR = '../data/processed/'
SAVED_MODEL   = '../ml_model/saved_model/'
RESULTS_DIR   = '../results/plots/'
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_models():
    svm = joblib.load(os.path.join(SAVED_MODEL, 'svm_model.pkl'))
    rf  = joblib.load(os.path.join(SAVED_MODEL, 'rf_model.pkl'))
    indices = joblib.load(os.path.join(SAVED_MODEL, 'feature_indices.pkl'))
    print("[OK] Models loaded.")
    return svm, rf, indices


def load_test_data():
    for name in ['sdn_ddos', 'nslkdd']:
        path = os.path.join(PROCESSED_DIR, f'{name}_processed.npz')
        if os.path.exists(path):
            data = np.load(path)
            X, y = data['X'], data['y']
            _, X_test, _, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            print(f"[OK] Test data loaded from {name}.")
            return X_test, y_test, name
    raise FileNotFoundError("No processed data found. Run preprocess.py first.")


def hybrid_predict(svm, rf, X, indices):
    X_sel = X[:, indices]
    svm_proba    = svm.predict_proba(X_sel)
    rf_proba     = rf.predict_proba(X_sel)
    avg_proba    = (svm_proba + rf_proba) / 2.0
    hybrid_preds = np.argmax(avg_proba, axis=1)
    return hybrid_preds, avg_proba[:, 1]   # preds + positive-class probability


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Normal', 'DDoS'],
        yticklabels=['Normal', 'DDoS']
    )
    plt.title('Confusion Matrix — Hybrid SVM-RF', fontsize=13, fontweight='bold')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_roc(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc      = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0,1],[0,1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve — Hybrid SVM-RF', fontsize=13, fontweight='bold')
    plt.legend(loc='lower right')
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'roc_curve.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")
    return roc_auc


def plot_accuracy_comparison(svm_acc, rf_acc, hybrid_acc):
    models = ['SVM', 'Random Forest', 'Hybrid SVM-RF']
    accs   = [svm_acc*100, rf_acc*100, hybrid_acc*100]
    colors = ['#4A90D9', '#27AE60', '#E74C3C']

    plt.figure(figsize=(7, 5))
    bars = plt.bar(models, accs, color=colors, width=0.5)
    for bar, val in zip(bars, accs):
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() - 1.5,
            f'{val:.2f}%', ha='center', va='top',
            color='white', fontsize=12, fontweight='bold'
        )
    plt.ylim([90, 101])
    plt.ylabel('Accuracy (%)')
    plt.title('Model Accuracy Comparison', fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'accuracy_comparison.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


if __name__ == '__main__':
    print("=" * 50)
    print(" DDoS Detection — Evaluation")
    print("=" * 50)

    svm, rf, indices = load_models()
    X_test, y_test, dataset_name = load_test_data()

    preds, scores = hybrid_predict(svm, rf, X_test, indices)
    X_sel = X_test[:, indices]

    # Individual model accuracies
    svm_acc    = accuracy_score(y_test, svm.predict(X_sel))
    rf_acc     = accuracy_score(y_test, rf.predict(X_sel))
    hybrid_acc = accuracy_score(y_test, preds)
    f1         = f1_score(y_test, preds, average='weighted')

    # False Alarm Rate (FPR for class 0)
    cm  = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0,0], 0, 0, cm[1,1])
    far = fp / (fp + tn) if (fp + tn) > 0 else 0

    print("\n── Results Summary ─────────────────────────────")
    print(f"  Dataset        : {dataset_name}")
    print(f"  SVM Accuracy   : {svm_acc*100:.2f}%")
    print(f"  RF  Accuracy   : {rf_acc*100:.2f}%")
    print(f"  Hybrid Accuracy: {hybrid_acc*100:.2f}%")
    print(f"  F1-Score       : {f1:.4f}")
    print(f"  False Alarm Rate: {far*100:.4f}%")
    print(f"\n{classification_report(y_test, preds, target_names=['Normal','DDoS'])}")

    print("\n── Generating Plots ────────────────────────────")
    plot_confusion_matrix(y_test, preds)
    roc_auc = plot_roc(y_test, scores)
    print(f"  AUC Score: {roc_auc:.4f}")
    plot_accuracy_comparison(svm_acc, rf_acc, hybrid_acc)

    print("\n[DONE] All plots saved to results/plots/")
