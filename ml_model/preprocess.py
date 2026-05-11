"""
preprocess.py
─────────────
Cleans and normalizes the SDN-DDoS and NSL-KDD datasets.
Run this FIRST before training the model.

Usage:
    python3 preprocess.py
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os, joblib

# ── Paths ──────────────────────────────────────────────────
RAW_DIR       = '../data/raw/'
PROCESSED_DIR = '../data/processed/'
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ── Column name for the label (attack vs normal) ────────────
#    Adjust this if your dataset uses a different column name.
LABEL_COLUMN  = 'label'


def load_sdn_ddos():
    path = os.path.join(RAW_DIR, 'sdn_ddos.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n[ERROR] File not found: {path}\n"
            "Download from: https://data.mendeley.com/datasets/b7vw628825/1\n"
            "Save it as data/raw/sdn_ddos.csv"
        )
    df = pd.read_csv(path)
    print(f"[SDN-DDoS] Loaded {len(df)} rows, {len(df.columns)} columns.")
    return df


def load_nslkdd():
    """NSL-KDD has no header row — columns are added manually."""
    path = os.path.join(RAW_DIR, 'KDDTrain+.txt')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n[ERROR] File not found: {path}\n"
            "Download from: https://www.unb.ca/cic/datasets/nsl.html\n"
            "Save KDDTrain+.txt in data/raw/"
        )
    cols = [
        'duration','protocol_type','service','flag','src_bytes','dst_bytes',
        'land','wrong_fragment','urgent','hot','num_failed_logins','logged_in',
        'num_compromised','root_shell','su_attempted','num_root',
        'num_file_creations','num_shells','num_access_files','num_outbound_cmds',
        'is_host_login','is_guest_login','count','srv_count','serror_rate',
        'srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate',
        'diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count',
        'dst_host_same_srv_rate','dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
        'dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate',
        'dst_host_srv_rerror_rate','label','difficulty'
    ]
    df = pd.read_csv(path, names=cols)
    df.drop('difficulty', axis=1, inplace=True)  # not needed
    # Binary label: normal=0, attack=1
    df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)
    print(f"[NSL-KDD]  Loaded {len(df)} rows, {len(df.columns)} columns.")
    return df


def clean(df):
    before = len(df)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    print(f"  Removed {before - len(df)} duplicate/null rows. Remaining: {len(df)}")
    return df


def encode_categoricals(df):
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        if col != LABEL_COLUMN:
            df[col] = le.fit_transform(df[col].astype(str))
            print(f"  Encoded column: {col}")
    return df


def scale_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, '../ml_model/saved_model/scaler.pkl')
    print("  Scaler saved to ml_model/saved_model/scaler.pkl")
    return X_scaled


def preprocess_dataset(df, name):
    print(f"\n── Processing {name} ──────────────────────────")
    df = clean(df)
    df = encode_categoricals(df)

    if LABEL_COLUMN not in df.columns:
        raise ValueError(
            f"Label column '{LABEL_COLUMN}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    X = df.drop(LABEL_COLUMN, axis=1)
    y = df[LABEL_COLUMN]

    print(f"  Features: {X.shape[1]} | Samples: {X.shape[0]}")
    print(f"  Label distribution:\n{y.value_counts().to_string()}")

    X_scaled = scale_features(X.values)

    # Save
    out_path = os.path.join(PROCESSED_DIR, f'{name}_processed.npz')
    np.savez(out_path, X=X_scaled, y=y.values)
    print(f"  Saved → {out_path}")
    return X_scaled, y.values


if __name__ == '__main__':
    print("=" * 50)
    print(" DDoS Detection — Data Preprocessing")
    print("=" * 50)

    try:
        df_sdn = load_sdn_ddos()
        preprocess_dataset(df_sdn, 'sdn_ddos')
    except FileNotFoundError as e:
        print(e)

    try:
        df_nsl = load_nslkdd()
        preprocess_dataset(df_nsl, 'nslkdd')
    except FileNotFoundError as e:
        print(e)

    print("\n[DONE] Preprocessing complete. Run train_model.py next.")
