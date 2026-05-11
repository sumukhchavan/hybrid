"""
train_model.py
──────────────
Trains the Hybrid SVM-RF model for DDoS detection.

Pipeline:
    1. Load preprocessed data
    2. RF selects top features
    3. SVM trains on selected features
    4. RF also trains (for ensemble voting)
    5. Both models saved to saved_model/

Usage:
    python3 train_model.py
"""

import numpy as np
import joblib, os, time
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report
from feature_selection import select_features

PROCESSED_DIR = '../data/processed/'
SAVED_MODEL   = '../ml_model/saved_model/'
os.makedirs(SAVED_MODEL, exist_ok=True)


def load_data(name='sdn_ddos'):
    path = os.path.join(PROCESSED_DIR, f'{name}_processed.npz')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n[ERROR] Processed data not found: {path}\n"
            "Run preprocess.py first:\n"
            "    python3 preprocess.py"
        )
    data = np.load(path)
    X, y = data['X'], data['y']
    print(f"[DATA] Loaded {name}: X={X.shape}, y={y.shape}")
    print(f"       Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    return X, y


def train_hybrid(X, y):
    # ── Feature selection via RF ─────────────────────────────
    X_selected, indices = select_features(X, y, top_n=15)

    # ── Train/test split ─────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n── Train/Test Split ────────────────────────────")
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Test:  {X_test.shape[0]} samples")

    # ── Train SVM ────────────────────────────────────────────
    print("\n── Training SVM (RBF kernel) ───────────────────")
    t0 = time.time()
    svm = SVC(
        kernel='rbf',
        C=10.0,
        gamma='scale',
        probability=True,
        random_state=42
    )
    svm.fit(X_train, y_train)
    print(f"  SVM trained in {time.time()-t0:.1f}s")

    # ── Train RF ─────────────────────────────────────────────
    print("\n── Training Random Forest ──────────────────────")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    print(f"  RF trained in {time.time()-t0:.1f}s")

    # ── Hybrid prediction (majority voting) ──────────────────
    svm_proba = svm.predict_proba(X_test)
    rf_proba  = rf.predict_proba(X_test)

    # Average probabilities from both models
    avg_proba     = (svm_proba + rf_proba) / 2.0
    hybrid_preds  = np.argmax(avg_proba, axis=1)

    # ── Results ───────────────────────────────────────────────
    print("\n── Evaluation Results ──────────────────────────")
    acc    = accuracy_score(y_test, hybrid_preds)
    f1     = f1_score(y_test, hybrid_preds, average='weighted')
    svm_acc = accuracy_score(y_test, svm.predict(X_test))
    rf_acc  = accuracy_score(y_test, rf.predict(X_test))

    print(f"  SVM Accuracy    : {svm_acc*100:.2f}%")
    print(f"  RF  Accuracy    : {rf_acc*100:.2f}%")
    print(f"  HYBRID Accuracy : {acc*100:.2f}%  ← Final")
    print(f"  Hybrid F1-Score : {f1:.4f}")
    print(f"\n{classification_report(y_test, hybrid_preds, target_names=['Normal','DDoS'])}")

    # ── Save models ───────────────────────────────────────────
    joblib.dump(svm, os.path.join(SAVED_MODEL, 'svm_model.pkl'))
    joblib.dump(rf,  os.path.join(SAVED_MODEL, 'rf_model.pkl'))
    print("  Models saved:")
    print("    → ml_model/saved_model/svm_model.pkl")
    print("    → ml_model/saved_model/rf_model.pkl")

    return svm, rf, X_test, y_test, hybrid_preds


if __name__ == '__main__':
    print("=" * 50)
    print(" DDoS Detection — Model Training")
    print("=" * 50)

    # Try SDN-DDoS dataset first, fall back to NSL-KDD
    try:
        X, y = load_data('sdn_ddos')
    except FileNotFoundError:
        try:
            X, y = load_data('nslkdd')
        except FileNotFoundError as e:
            print(e)
            exit(1)

    svm, rf, X_test, y_test, preds = train_hybrid(X, y)
    print("\n[DONE] Training complete. Run evaluate.py to see full plots.")
