"""
hybrid_detector.py
──────────────────
Core inference module. Loads saved SVM and RF models and
returns a hybrid prediction for a given traffic flow sample.

Used by the Ryu controller in real-time detection.

Usage:
    from detection.hybrid_detector import HybridDetector
    detector = HybridDetector()
    result   = detector.predict(flow_features)
"""

import numpy as np
import joblib, os

SAVED_MODEL = '../ml_model/saved_model/'


class HybridDetector:
    """
    Loads trained SVM and RF models and provides hybrid prediction.

    Prediction logic:
        - Both models output class probabilities.
        - Probabilities are averaged (soft voting).
        - Final label = argmax of averaged probabilities.
        - If confidence < threshold, flag as uncertain.
    """

    CONFIDENCE_THRESHOLD = 0.70   # below this = uncertain

    def __init__(self, model_dir=SAVED_MODEL):
        svm_path     = os.path.join(model_dir, 'svm_model.pkl')
        rf_path      = os.path.join(model_dir, 'rf_model.pkl')
        indices_path = os.path.join(model_dir, 'feature_indices.pkl')
        scaler_path  = os.path.join(model_dir, 'scaler.pkl')

        if not all(os.path.exists(p) for p in [svm_path, rf_path, indices_path]):
            raise FileNotFoundError(
                "Saved models not found. Run ml_model/train_model.py first."
            )

        self.svm     = joblib.load(svm_path)
        self.rf      = joblib.load(rf_path)
        self.indices = joblib.load(indices_path)
        self.scaler  = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

        print("[HybridDetector] Models loaded successfully.")

    def predict(self, flow_features):
        """
        Predict whether a flow is DDoS or Normal.

        Parameters:
            flow_features : list or numpy array of raw feature values
                            (must match the original feature dimensionality)

        Returns:
            dict with keys:
              - 'label'      : 'DDoS' or 'Normal'
              - 'confidence' : float (0.0 to 1.0)
              - 'is_attack'  : bool
        """
        X = np.array(flow_features).reshape(1, -1)

        # Scale if scaler is available
        if self.scaler:
            X = self.scaler.transform(X)

        # Select only trained features
        X_sel = X[:, self.indices]

        # Soft voting
        svm_proba = self.svm.predict_proba(X_sel)[0]
        rf_proba  = self.rf.predict_proba(X_sel)[0]
        avg_proba = (svm_proba + rf_proba) / 2.0

        label_idx  = int(np.argmax(avg_proba))
        confidence = float(avg_proba[label_idx])
        label      = 'DDoS' if label_idx == 1 else 'Normal'

        return {
            'label'     : label,
            'confidence': round(confidence, 4),
            'is_attack' : label_idx == 1,
            'uncertain' : confidence < self.CONFIDENCE_THRESHOLD
        }

    def predict_batch(self, flows_matrix):
        """
        Predict for multiple flows at once (used by Ryu controller).

        Parameters:
            flows_matrix : 2D numpy array (n_flows x n_features)

        Returns:
            list of dicts (one per flow)
        """
        X = np.array(flows_matrix)
        if self.scaler:
            X = self.scaler.transform(X)
        X_sel     = X[:, self.indices]
        svm_proba = self.svm.predict_proba(X_sel)
        rf_proba  = self.rf.predict_proba(X_sel)
        avg_proba = (svm_proba + rf_proba) / 2.0

        results = []
        for proba in avg_proba:
            idx  = int(np.argmax(proba))
            conf = float(proba[idx])
            results.append({
                'label'     : 'DDoS' if idx == 1 else 'Normal',
                'confidence': round(conf, 4),
                'is_attack' : idx == 1,
                'uncertain' : conf < self.CONFIDENCE_THRESHOLD
            })
        return results


# ── Quick test ────────────────────────────────────────────────
if __name__ == '__main__':
    print("Testing HybridDetector with dummy data...")
    try:
        detector = HybridDetector()
        dummy    = np.random.rand(30).tolist()   # 30 raw features
        result   = detector.predict(dummy)
        print(f"Prediction: {result}")
    except FileNotFoundError as e:
        print(e)
