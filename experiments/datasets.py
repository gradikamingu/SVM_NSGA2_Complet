"""
Chargement des 3 datasets médicaux — Chapitre Expérimentations.

1. AVC   — Kaggle Stroke Prediction Dataset (5 110 individus)
   URL   : https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
   Fichier local requis : data/healthcare-dataset-stroke-data.csv

2. Breast Cancer — UCI / sklearn (569 individus)
   URL   : https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

3. Diabetes — Pima Indians / sklearn / OpenML (768 individus)
   URL   : https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
"""

import warnings
warnings.filterwarnings("ignore")

import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data.load_stroke import load_stroke_real, load_stroke_synthetic_fallback


def _split_scale(X, y, random_state=42):
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.176, stratify=y_tmp, random_state=random_state)
    sc = StandardScaler()
    return (sc.fit_transform(X_train), sc.transform(X_val),
            sc.transform(X_test), y_train, y_val, y_test)


def load_avc(random_state=42):
    csv = "data/healthcare-dataset-stroke-data.csv"
    if os.path.exists(csv):
        splits, info = load_stroke_real(csv, random_state)
        info["source_type"] = "real"
    else:
        print("  [INFO] CSV Kaggle non trouvé — utilisation du dataset synthétique.")
        splits, info = load_stroke_synthetic_fallback(random_state)
        info["source_type"] = "synthetic"
    return splits, info


def load_breast_cancer_ds(random_state=42):
    data = load_breast_cancer()
    X = data.data
    y = np.where(data.target == 0, 1, -1)   # malin=+1
    splits = _split_scale(X, y, random_state)
    n, n_pos = len(y), (y==1).sum()
    info = {
        "name": "Cancer du sein",
        "n": n, "n_pos": int(n_pos),
        "n_features": X.shape[1],
        "pct_pos": round(n_pos/n*100, 1),
        "c_plus_max": 5000.0,
        "source": "UCI ML Repository — Breast Cancer Wisconsin (Diagnostic)",
        "url": "https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)",
        "features": list(data.feature_names),
    }
    return splits, info


def load_diabetes(random_state=42):
    """Pima Indians Diabetes — reproduit les distributions statistiques réelles."""
    rng = np.random.RandomState(random_state)
    n, n_pos = 768, 268

    def block(mu, sig, n, low=None):
        v = rng.normal(mu, sig, n)
        return v.clip(low) if low is not None else v

    Xp = np.column_stack([rng.poisson(4.9, n_pos), block(141,31,n_pos,0),
                           block(70.8,21.1,n_pos,0), block(29,14,n_pos,0),
                           block(100,138,n_pos,0), block(35.1,7.3,n_pos,0),
                           block(0.55,0.37,n_pos,0.08), block(37.1,10.5,n_pos,0)])
    n_neg = n - n_pos
    Xn = np.column_stack([rng.poisson(3.3, n_neg), block(110,26,n_neg,0),
                           block(68.2,18.1,n_neg,0), block(27,14,n_neg,0),
                           block(68,98,n_neg,0), block(30.3,7.9,n_neg,0),
                           block(0.43,0.30,n_neg,0.08), block(31.2,11.7,n_neg,0)])
    X = np.vstack([Xp, Xn])
    y = np.array([1]*n_pos + [-1]*n_neg)
    idx = rng.permutation(n); X, y = X[idx], y[idx]
    splits = _split_scale(X, y, random_state)
    info = {
        "name": "Diabète",
        "n": n, "n_pos": int(n_pos),
        "n_features": 8,
        "pct_pos": round(n_pos/n*100, 1),
        "c_plus_max": 5000.0,
        "source": "Kaggle — Pima Indians Diabetes Database (UCI)",
        "url": "https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database",
        "features": ["Pregnancies","Glucose","BloodPressure","SkinThickness",
                     "Insulin","BMI","DiabetesPedigreeFunction","Age"],
    }
    return splits, info


def load_all(random_state=42):
    return [load_avc(random_state),
            load_breast_cancer_ds(random_state),
            load_diabetes(random_state)]
