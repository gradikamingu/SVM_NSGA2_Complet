"""
Génération du dataset synthétique pour la prédiction binaire du risque d'AVC.

Ce dataset simule les caractéristiques du "Stroke Prediction Dataset" (Kaggle),
basé sur des paramètres médicaux réels identifiés dans la littérature.

Caractéristiques utilisées :
    - age              : âge du patient (années)
    - hypertension     : hypertension artérielle (0/1)
    - heart_disease    : maladie cardiaque (0/1)
    - avg_glucose_level: taux moyen de glucose (mg/dL)
    - bmi              : indice de masse corporelle
    - smoking_status   : statut tabagique (encodé)
    - ever_married     : marié(e) (0/1)
    - work_type        : type de travail (encodé)

Cible :
    - stroke : 1 = risque d'AVC, -1 = pas de risque (convention SVM)

Le dataset est volontairement déséquilibré (≈15% de positifs)
pour refléter la réalité clinique et justifier l'usage de C+/C-.
"""

import numpy as np
import pandas as pd

def generate_stroke_dataset(n_samples=1000, positive_rate=0.15, random_state=42):
    rng = np.random.RandomState(random_state)

    n_pos = int(n_samples * positive_rate)
    n_neg = n_samples - n_pos

    def sample_features(n, is_positive):
        """Génère des features corrélées au risque d'AVC."""
        # Âge : les AVC touchent plus les personnes âgées
        if is_positive:
            age = rng.normal(loc=68, scale=12, size=n).clip(30, 95)
        else:
            age = rng.normal(loc=45, scale=15, size=n).clip(18, 90)

        # Hypertension : plus fréquente chez les patients AVC
        hypertension = rng.binomial(1, 0.55 if is_positive else 0.20, n)

        # Maladie cardiaque
        heart_disease = rng.binomial(1, 0.35 if is_positive else 0.08, n)

        # Glycémie à jeun (mg/dL)
        if is_positive:
            glucose = rng.normal(loc=145, scale=45, size=n).clip(60, 300)
        else:
            glucose = rng.normal(loc=95, scale=25, size=n).clip(60, 200)

        # IMC
        bmi = rng.normal(loc=30 if is_positive else 26, scale=6, size=n).clip(15, 55)

        # Tabagisme : 0=jamais, 1=ancien, 2=fumeur actuel
        if is_positive:
            smoking = rng.choice([0, 1, 2], n, p=[0.25, 0.40, 0.35])
        else:
            smoking = rng.choice([0, 1, 2], n, p=[0.40, 0.35, 0.25])

        # Marié(e)
        ever_married = rng.binomial(1, 0.75 if is_positive else 0.55, n)

        # Type de travail : 0=privé, 1=indépendant, 2=gouvernement, 3=sans emploi
        work_type = rng.choice([0, 1, 2, 3], n, p=[0.45, 0.25, 0.20, 0.10])

        return np.column_stack([
            age, hypertension, heart_disease, glucose, bmi,
            smoking, ever_married, work_type
        ])

    X_pos = sample_features(n_pos, is_positive=True)
    X_neg = sample_features(n_neg, is_positive=False)

    X = np.vstack([X_pos, X_neg])
    # Convention SVM : +1 pour positif (AVC), -1 pour négatif
    y = np.array([1] * n_pos + [-1] * n_neg)

    # Mélange
    idx = rng.permutation(n_samples)
    X, y = X[idx], y[idx]

    columns = [
        'age', 'hypertension', 'heart_disease',
        'avg_glucose_level', 'bmi',
        'smoking_status', 'ever_married', 'work_type'
    ]
    df = pd.DataFrame(X, columns=columns)
    df['stroke'] = y

    return df


if __name__ == "__main__":
    df = generate_stroke_dataset(n_samples=1000)
    df.to_csv("stroke_dataset.csv", index=False)
    print(f"Dataset généré : {df.shape[0]} individus")
    print(f"  Positifs (AVC)    : {(df['stroke'] == 1).sum()} ({(df['stroke'] == 1).mean():.1%})")
    print(f"  Négatifs (no AVC) : {(df['stroke'] == -1).sum()} ({(df['stroke'] == -1).mean():.1%})")
    print(df.describe().round(2))
