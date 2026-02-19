"""
Chargement et prétraitement du dataset AVC réel.

Source : Kaggle — Stroke Prediction Dataset
Lien   : https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
Fichier: healthcare-dataset-stroke-data.csv
         (à placer dans data/ avant l'exécution)

Statistiques réelles du dataset :
    - 5 110 individus
    - 248 cas AVC (4,87 %)
    - 11 features (après suppression de id)
    - Valeurs manquantes dans bmi (~201 lignes)
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def _encode(df):
    """Encodage des variables catégorielles."""
    # Gender : Male=1, Female=0, Other → supprimé
    df = df[df["gender"] != "Other"].copy()
    df["gender"] = (df["gender"] == "Male").astype(int)

    # ever_married : Yes=1, No=0
    df["ever_married"] = (df["ever_married"] == "Yes").astype(int)

    # work_type : encodage ordinal sémantique
    work_map = {"children": 0, "Never_worked": 1, "Govt_job": 2,
                "Private": 3, "Self-employed": 4}
    df["work_type"] = df["work_type"].map(work_map).fillna(3)

    # Residence_type : Urban=1, Rural=0
    df["Residence_type"] = (df["Residence_type"] == "Urban").astype(int)

    # smoking_status : encodage ordinal
    smoke_map = {"Unknown": 0, "never smoked": 1,
                 "formerly smoked": 2, "smokes": 3}
    df["smoking_status"] = df["smoking_status"].map(smoke_map).fillna(0)

    return df


def load_stroke_real(csv_path="data/healthcare-dataset-stroke-data.csv",
                     random_state=42):
    """
    Charge le vrai dataset AVC depuis le CSV Kaggle.
    Retourne : splits (X_train, X_val, X_test, y_train, y_val, y_test), info
    """
    df = pd.read_csv(csv_path)

    # Nettoyage
    df = df.drop(columns=["id"], errors="ignore")
    df = df[df["gender"] != "Other"].copy()

    # BMI manquant → médiane
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
    df["bmi"] = df["bmi"].fillna(df["bmi"].median())

    # Encodage catégoriel
    df = _encode(df)

    # Cible en convention SVM : stroke=1 → +1, stroke=0 → -1
    X = df.drop(columns=["stroke"]).values.astype(float)
    y = np.where(df["stroke"].values == 1, 1, -1)

    n = len(y)
    n_pos = (y == 1).sum()
    pct_pos = round(n_pos / n * 100, 2)

    # Split stratifié 70/15/15
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.176, stratify=y_tmp, random_state=random_state)

    # Normalisation
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    splits = (X_train, X_val, X_test, y_train, y_val, y_test)
    info = {
        "name": "AVC",
        "n": n, "n_pos": int(n_pos),
        "n_features": X.shape[1],
        "pct_pos": pct_pos,
        "c_plus_max": 5000.0,
        "source": "Kaggle — Stroke Prediction Dataset",
        "url": "https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset",
        "features": list(df.drop(columns=["stroke"]).columns),
    }
    return splits, info


def load_stroke_synthetic_fallback(random_state=42):
    """
    Dataset AVC synthétique de secours si le CSV n'est pas disponible.
    Reproduit les distributions statistiques du vrai dataset Kaggle.
    """
    from data.generate_dataset import generate_stroke_dataset
    df = generate_stroke_dataset(n_samples=5110, positive_rate=0.0487,
                                 random_state=random_state)
    X = df.drop(columns=["stroke"]).values
    y = df["stroke"].values

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.176, stratify=y_tmp, random_state=random_state)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    n, n_pos = len(y), (y==1).sum()
    splits = (X_train, X_val, X_test, y_train, y_val, y_test)
    info = {
        "name": "AVC",
        "n": n, "n_pos": int(n_pos),
        "n_features": X.shape[1],
        "pct_pos": round(n_pos/n*100, 2),
        "c_plus_max": 5000.0,
        "source": "Synthétique (distributions Kaggle)",
        "url": "https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset",
        "features": ["age","hypertension","heart_disease","avg_glucose_level",
                     "bmi","smoking_status","ever_married","work_type"],
    }
    return splits, info
