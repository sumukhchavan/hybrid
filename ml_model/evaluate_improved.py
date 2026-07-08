"""
evaluate_improved.py  —  Improved Evaluation & Plot Generation
──────────────────────────────────────────────────────────────
Improvements over original evaluate.py:
  ✓ Loads exact test split saved during training (consistent results)
  ✓ Adds Precision-Recall curve plot
  ✓ Improved plot styling (larger fonts, gridlines, better colors)
  ✓ Prints Detection Rate and False Alarm Rate
  ✓ Saves a text summary report alongside plots

Usage:
    cd ml_model
    python evaluate_improved.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend for saving files
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, os, sys
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, f1_score,
    roc_curve, auc,
    precision_recall_curve, average_precision_score
)

SAVED_MODEL = os.path.join(os.path.dirname(__file__), 'saved_model')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'plots')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Consistent matplotlib style
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
})


def load_models():
    svm     = joblib.load(os.path.join(SAVED_MODEL, 'svm_model.pkl'))
    rf      = joblib.load(os.path.join(SAVED_MODEL, 'rf_model.pkl'))
    indices = joblib.load(os.path.join(SAVED_MODEL, 'feature_indices.pkl'))
    print("[OK] SVM, RF models and feature indices loaded.")
    return svm, rf, indices


def load_test_split():
    """Load the exact test split saved during training."""
    split_path = os.path.join(SAVED_MODEL, 'test_split.npz')
    if os.path.exists(split_path):
        data = np.load(split_path)
        print("[OK] Loaded saved test split (consistent with training).")
        # If scores were pre-computed and saved, use them; else recompute
        scores = data.get('scores', None)
        return data['X_test'], data['y_test'], scores
    # Fallback to loading processed data
    from sklearn.model_selection import train_test_split
    PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    for name in ['sdn_ddos', 'nslkdd']:
        path = os.path.join(PROCESSED_DIR, f'{name}_processed.npz')
        if os.path.exists(path):
            data = np.load(path)
            X, y = data['X'], data['y']
            _, X_test, _, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y)
            print(f"[OK] Test data loaded from {name} (fallback).")
            return X_test, y_test, None
    raise FileNotFoundError("No test data found. Run train_improved.py first.")


def hybrid_predict(svm, rf, X, indices):
    X_sel     = X[:, indices]
    svm_proba = svm.predict_proba(X_sel)
    rf_proba  = rf.predict_proba(X_sel)
    avg_proba = (svm_proba + rf_proba) / 2.0
    preds     = np.argmax(avg_proba, axis=1)
    return preds, avg_proba[:, 1], X_sel


# ── Plot 1: Confusion Matrix ──────────────────────────────────
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'DDoS'],
                yticklabels=['Normal', 'DDoS'],
                linewidths=0.5, linecolor='gray',
                annot_kws={'size': 16, 'weight': 'bold'},
                ax=ax)
    ax.set_title('Confusion Matrix — Hybrid SVM-RF',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('Actual Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.tick_params(labelsize=11)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: confusion_matrix.png")
    return path


# ── Plot 2: ROC Curve ─────────────────────────────────────────
def plot_roc(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc     = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='#E74C3C', lw=2.5,
            label=f'Hybrid SVM-RF  (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='#95A5A6', lw=1.5,
            linestyle='--', label='Random Chance')
    ax.fill_between(fpr, tpr, alpha=0.08, color='#E74C3C')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve — Hybrid SVM-RF',
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='lower right', fontsize=11)
    ax.tick_params(labelsize=11)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'roc_curve.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: roc_curve.png   (AUC={roc_auc:.4f})")
    return roc_auc


# ── Plot 3: Precision-Recall Curve (NEW) ─────────────────────
def plot_precision_recall(y_true, y_scores):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color='#27AE60', lw=2.5,
            label=f'Hybrid SVM-RF  (AP = {ap:.4f})')
    ax.fill_between(recall, precision, alpha=0.08, color='#27AE60')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve — Hybrid SVM-RF',
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='lower left', fontsize=11)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.tick_params(labelsize=11)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'precision_recall_curve.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: precision_recall_curve.png  (AP={ap:.4f})")
    return ap


# ── Plot 4: Accuracy Comparison ───────────────────────────────
def plot_accuracy_comparison(svm_acc, rf_acc, hybrid_acc):
    models = ['SVM\n(RBF, C=10)', 'Random Forest\n(200 trees)', 'Hybrid\nSVM-RF']
    accs   = [svm_acc * 100, rf_acc * 100, hybrid_acc * 100]
    colors = ['#4A90D9', '#27AE60', '#E74C3C']

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(models, accs, color=colors, width=0.45,
                  edgecolor='white', linewidth=1.2)
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() - 0.6,
                f'{val:.2f}%', ha='center', va='top',
                color='white', fontsize=13, fontweight='bold')
    lo = min(accs) - 2
    ax.set_ylim([max(lo, 88), 101])
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Model Accuracy Comparison', fontsize=14,
                 fontweight='bold', pad=15)
    ax.tick_params(labelsize=11)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'accuracy_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: accuracy_comparison.png")


# ── Text report ───────────────────────────────────────────────
def save_report(svm_acc, rf_acc, hybrid_acc, f1, roc_auc, ap,
                far, dr, y_test, preds):
    report_text = classification_report(
        y_test, preds, target_names=['Normal', 'DDoS'])
    lines = [
        "=" * 55,
        " DDoS DETECTION — HYBRID SVM-RF EVALUATION REPORT",
        "=" * 55,
        f"  SVM  Accuracy         : {svm_acc*100:.4f}%",
        f"  RF   Accuracy         : {rf_acc*100:.4f}%",
        f"  HYBRID Accuracy       : {hybrid_acc*100:.4f}%",
        f"  Hybrid F1-Score       : {f1:.4f}",
        f"  ROC-AUC Score         : {roc_auc:.4f}",
        f"  Average Precision     : {ap:.4f}",
        f"  Detection Rate (DR)   : {dr*100:.4f}%",
        f"  False Alarm Rate (FAR): {far*100:.4f}%",
        "=" * 55,
        "",
        "Classification Report:",
        report_text,
    ]
    path = os.path.join(RESULTS_DIR, 'evaluation_report.txt')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  ✓ Saved: evaluation_report.txt")
    # Also print to console
    print('\n' + '\n'.join(lines))


if __name__ == '__main__':
    print("=" * 55)
    print(" DDoS Detection — Improved Evaluation")
    print("=" * 55)

    svm, rf, indices = load_models()
    X_test_raw, y_test, saved_scores = load_test_split()

    preds, scores, X_sel = hybrid_predict(svm, rf, X_test_raw, indices)

    # Use saved scores if available (from training), else use freshly computed
    if saved_scores is not None and len(saved_scores) == len(y_test):
        scores = saved_scores
        print("[INFO] Using pre-computed probability scores from training.")

    svm_acc    = accuracy_score(y_test, svm.predict(X_sel))
    rf_acc     = accuracy_score(y_test, rf.predict(X_sel))
    hybrid_acc = accuracy_score(y_test, preds)
    f1         = f1_score(y_test, preds, average='weighted')

    cm = confusion_matrix(y_test, preds)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        dr  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    else:
        far = dr = float('nan')

    print(f"\n  SVM  Accuracy  : {svm_acc*100:.4f}%")
    print(f"  RF   Accuracy  : {rf_acc*100:.4f}%")
    print(f"  HYBRID Accuracy: {hybrid_acc*100:.4f}%")

    print("\n── Generating Plots ─────────────────────────────")
    plot_confusion_matrix(y_test, preds)
    roc_auc = plot_roc(y_test, scores)
    ap      = plot_precision_recall(y_test, scores)
    plot_accuracy_comparison(svm_acc, rf_acc, hybrid_acc)
    save_report(svm_acc, rf_acc, hybrid_acc, f1, roc_auc, ap,
                far, dr, y_test, preds)

    print(f"\n[DONE] All outputs saved to: results/plots/")
