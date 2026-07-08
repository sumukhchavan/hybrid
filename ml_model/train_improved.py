"""
train_improved.py  —  Improved Hybrid SVM-RF Training (Large-Scale)
────────────────────────────────────────────────────────────────────
Improvements over original train_model.py:
  ✓ Uses LinearSVC + CalibratedClassifierCV (scales to 1M+ samples)
    instead of kernel SVC which is O(n²) memory and too slow at this scale
  ✓ 5-fold stratified cross-validation for both SVM and RF
  ✓ ROC-AUC metric reported
  ✓ Detection Rate (DR) and False Alarm Rate (FAR) computed
  ✓ RF n_estimators raised to 200 for better accuracy
  ✓ Test split saved so evaluate_improved.py uses identical data
  ✓ Prints training time per model

Usage:
    cd ml_model
    python train_improved.py
"""

import numpy as np
import joblib, os, sys, time
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold
)
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    roc_auc_score, confusion_matrix
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_selection import select_features

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
SAVED_MODEL   = os.path.join(os.path.dirname(__file__), 'saved_model')
os.makedirs(SAVED_MODEL, exist_ok=True)


def load_data(name='sdn_ddos'):
    path = os.path.join(PROCESSED_DIR, f'{name}_processed.npz')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n[ERROR] Processed data not found: {path}\n"
            "Run preprocess.py first."
        )
    data = np.load(path)
    X, y = data['X'], data['y']
    print(f"[DATA] Loaded '{name}': X={X.shape}, y={y.shape}")
    unique, counts = np.unique(y, return_counts=True)
    dist = dict(zip(unique.tolist(), counts.tolist()))
    print(f"       Class distribution: {dist}")
    if len(counts) > 1:
        print(f"       Attack traffic : {counts[1]/counts.sum()*100:.1f}%")
    return X, y


def train_hybrid(X, y):
    # ── 1. Feature selection via Random Forest ────────────────
    X_selected, indices = select_features(X, y, top_n=15)

    # ── 2. Train / Test split ─────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n[SPLIT] Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ── 3. Train SVM (LinearSVC + Platt scaling for probabilities) ──
    # LinearSVC is O(n) — handles millions of samples efficiently.
    print("\n[SVM] Training LinearSVC (C=1.0) ...")
    t0 = time.time()
    base_svm = LinearSVC(C=1.0, max_iter=2000, random_state=42)
    svm = CalibratedClassifierCV(base_svm, cv=5, method='sigmoid')
    svm.fit(X_train, y_train)
    svm_time = time.time() - t0
    print(f"  Trained in {svm_time:.2f}s")

    print("  5-fold cross-validation ...")
    svm_cv = cross_val_score(svm, X_train, y_train, cv=cv,
                             scoring='accuracy', n_jobs=-1)
    print(f"  CV Accuracy: {svm_cv.mean()*100:.2f}% +/- {svm_cv.std()*100:.2f}%")

    # ── 4. Train RF ───────────────────────────────────────────
    print("\n[RF] Training Random Forest (200 trees) ...")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_time = time.time() - t0
    print(f"  Trained in {rf_time:.2f}s")

    print("  5-fold cross-validation ...")
    rf_cv = cross_val_score(rf, X_train, y_train, cv=cv,
                            scoring='accuracy', n_jobs=-1)
    print(f"  CV Accuracy: {rf_cv.mean()*100:.2f}% +/- {rf_cv.std()*100:.2f}%")

    # ── 5. Hybrid soft-voting ─────────────────────────────────
    svm_proba    = svm.predict_proba(X_test)
    rf_proba     = rf.predict_proba(X_test)
    avg_proba    = (svm_proba + rf_proba) / 2.0
    hybrid_preds = np.argmax(avg_proba, axis=1)
    pos_scores   = avg_proba[:, 1]

    # ── 6. Compute all metrics ────────────────────────────────
    svm_acc    = accuracy_score(y_test, svm.predict(X_test))
    rf_acc     = accuracy_score(y_test, rf.predict(X_test))
    hybrid_acc = accuracy_score(y_test, hybrid_preds)
    f1         = f1_score(y_test, hybrid_preds, average='weighted')
    try:
        roc_auc = roc_auc_score(y_test, pos_scores)
    except Exception:
        roc_auc = float('nan')

    cm = confusion_matrix(y_test, hybrid_preds)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        dr  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    else:
        far = dr = float('nan')

    print("\n" + "=" * 55)
    print(" RESULTS SUMMARY")
    print("=" * 55)
    print(f"  SVM  Accuracy         : {svm_acc*100:.4f}%")
    print(f"  RF   Accuracy         : {rf_acc*100:.4f}%")
    print(f"  HYBRID Accuracy       : {hybrid_acc*100:.4f}%  << Final")
    print(f"  Hybrid F1-Score       : {f1:.4f}")
    print(f"  ROC-AUC Score         : {roc_auc:.4f}")
    print(f"  Detection Rate (DR)   : {dr*100:.4f}%")
    print(f"  False Alarm Rate (FAR): {far*100:.4f}%")
    print(f"  SVM Training Time     : {svm_time:.2f}s")
    print(f"  RF  Training Time     : {rf_time:.2f}s")
    print(f"  SVM 5-Fold CV         : {svm_cv.mean()*100:.2f}% +/- {svm_cv.std()*100:.2f}%")
    print(f"  RF  5-Fold CV         : {rf_cv.mean()*100:.2f}% +/- {rf_cv.std()*100:.2f}%")
    print("=" * 55)
    print()
    print("Classification Report:")
    print(classification_report(y_test, hybrid_preds,
                                target_names=['Normal', 'DDoS']))

    # ── 7. Save models and test split ─────────────────────────
    joblib.dump(svm, os.path.join(SAVED_MODEL, 'svm_model.pkl'))
    joblib.dump(rf,  os.path.join(SAVED_MODEL, 'rf_model.pkl'))
    np.savez(os.path.join(SAVED_MODEL, 'test_split.npz'),
             X_test=X_test, y_test=y_test, scores=pos_scores)
    print("  [OK] svm_model.pkl  saved")
    print("  [OK] rf_model.pkl   saved")
    print("  [OK] test_split.npz saved")

    return svm, rf, X_test, y_test, hybrid_preds, pos_scores


if __name__ == '__main__':
    print("=" * 55)
    print(" DDoS Detection -- Improved Hybrid SVM-RF Training")
    print("=" * 55)

    try:
        X, y = load_data('sdn_ddos')
    except FileNotFoundError:
        try:
            X, y = load_data('nslkdd')
        except FileNotFoundError as e:
            print(e)
            sys.exit(1)

    train_hybrid(X, y)
    print("\n[DONE] Run evaluate_improved.py to generate all plots.")
