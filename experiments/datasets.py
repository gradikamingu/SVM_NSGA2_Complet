"""
Chargement et préparation de tous les datasets utilisés dans les expérimentations.

Datasets :
    1. AVC          — généré synthétiquement (dataset principal)
    2. Breast Cancer — UCI (via sklearn)
    3. Diabetes      — Pima Indians (UCI, fichier local ou sklearn)
    4. Fraud         — sous-ensemble synthétique représentatif (déséquilibre extrême)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data.generate_dataset import generate_stroke_dataset


def _split_and_scale(X, y, random_state=42):
    """
    Division stratifiée 70/15/15 + StandardScaler ajusté sur train.
    Retourne : X_train, X_val, X_test, y_train, y_val, y_test
    """
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.176, stratify=y_tmp, random_state=random_state)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test


def load_avc(random_state=42):
    """Dataset AVC synthétique — 1 000 individus, 15 % positifs."""
    df = generate_stroke_dataset(n_samples=1000, positive_rate=0.15,
                                 random_state=random_state)
    X = df.drop(columns=["stroke"]).values
    y = df["stroke"].values   # {+1, -1}
    splits = _split_and_scale(X, y, random_state)
    info = {"name": "AVC", "n": len(y),
            "n_features": X.shape[1],
            "pct_pos": round((y == 1).mean() * 100, 1),
            "c_plus_max": 5000.0}
    return splits, info


def load_breast_cancer_ds(random_state=42):
    """Breast Cancer Wisconsin — 569 individus, 37 % malins."""
    data = load_breast_cancer()
    X = data.data
    # sklearn : 1=bénin, 0=malin → on veut +1=malin (positif = maladie)
    y = np.where(data.target == 0, 1, -1)
    splits = _split_and_scale(X, y, random_state)
    info = {"name": "Breast Cancer", "n": len(y),
            "n_features": X.shape[1],
            "pct_pos": round((y == 1).mean() * 100, 1),
            "c_plus_max": 5000.0}
    return splits, info


def load_diabetes(random_state=42):
    """
    Pima Indians Diabetes — 768 individus, 35 % diabétiques.
    Généré synthétiquement avec les distributions statistiques réelles.
    """
    rng = np.random.RandomState(random_state)
    n = 768
    n_pos = 268   # diabétiques

    # Diabétiques (y = +1)
    glucose_p  = rng.normal(141, 31, n_pos)
    bmi_p      = rng.normal(35.1, 7.3, n_pos)
    age_p      = rng.normal(37.1, 10.5, n_pos)
    bp_p       = rng.normal(70.8, 21.1, n_pos)
    insulin_p  = rng.normal(100, 138, n_pos).clip(0)
    skinfold_p = rng.normal(29, 14, n_pos).clip(0)
    dpf_p      = rng.normal(0.55, 0.37, n_pos).clip(0.08)
    preg_p     = rng.poisson(4.9, n_pos)

    # Non-diabétiques (y = -1)
    n_neg = n - n_pos
    glucose_n  = rng.normal(110, 26, n_neg)
    bmi_n      = rng.normal(30.3, 7.9, n_neg)
    age_n      = rng.normal(31.2, 11.7, n_neg)
    bp_n       = rng.normal(68.2, 18.1, n_neg)
    insulin_n  = rng.normal(68, 98, n_neg).clip(0)
    skinfold_n = rng.normal(27, 14, n_neg).clip(0)
    dpf_n      = rng.normal(0.43, 0.30, n_neg).clip(0.08)
    preg_n     = rng.poisson(3.3, n_neg)

    def stack(a, b):
        return np.concatenate([a, b])

    X = np.column_stack([
        stack(preg_p, preg_n),
        stack(glucose_p, glucose_n),
        stack(bp_p, bp_n),
        stack(skinfold_p, skinfold_n),
        stack(insulin_p, insulin_n),
        stack(bmi_p, bmi_n),
        stack(dpf_p, dpf_n),
        stack(age_p, age_n),
    ])
    y = np.array([1] * n_pos + [-1] * n_neg)

    # Shuffle
    idx = rng.permutation(n)
    X, y = X[idx], y[idx]

    splits = _split_and_scale(X, y, random_state)
    info = {"name": "Diabetes", "n": n,
            "n_features": 8,
            "pct_pos": round(n_pos / n * 100, 1),
            "c_plus_max": 5000.0}
    return splits, info


def load_fraud(random_state=42):
    """
    Fraud synthétique — déséquilibre extrême (~0.5 % de fraudes).
    Simplifié à 3 000 transactions pour temps d'exécution raisonnable
    tout en conservant le ratio de déséquilibre.
    """
    rng = np.random.RandomState(random_state)
    n_total = 3000
    n_fraud = 15   # ~0.5 %
    n_legit = n_total - n_fraud
    n_feat  = 10   # features PCA-like

    # Légitimes — distribution standard
    X_legit = rng.randn(n_legit, n_feat)

    # Fraudes — légèrement décalées
    X_fraud = rng.randn(n_fraud, n_feat) + rng.uniform(0.5, 1.5, n_feat)

    X = np.vstack([X_legit, X_fraud])
    y = np.array([-1] * n_legit + [1] * n_fraud)

    idx = rng.permutation(n_total)
    X, y = X[idx], y[idx]

    splits = _split_and_scale(X, y, random_state)
    info = {"name": "Fraud", "n": n_total,
            "n_features": n_feat,
            "pct_pos": round(n_fraud / n_total * 100, 2),
            "c_plus_max": 50000.0}
    return splits, info


def load_all(random_state=42):
    """Charge les 4 datasets. Retourne une liste de (splits, info)."""
    return [
        load_avc(random_state),
        load_breast_cancer_ds(random_state),
        load_diabetes(random_state),
        load_fraud(random_state),
    ]
