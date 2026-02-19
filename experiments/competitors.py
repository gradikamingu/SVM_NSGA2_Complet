"""
Méthodes concurrentes pour la comparaison (Section 4.5 du chapitre).

Méthodes implémentées :
    1. SVM GridSearch + AUC (baseline, déjà dans main.py — réimplémentée ici)
    2. Random Forest avec class_weight='balanced'
    3. Gradient Boosting (équivalent XGBoost, disponible dans sklearn)
    4. Régression Logistique avec class_weight='balanced'
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


def _metrics(y_true, y_pred):
    """Calcule NFA, NFR et toutes les métriques depuis y_true / y_pred."""
    TP = int(np.sum((y_true == 1)  & (y_pred == 1)))
    FP = int(np.sum((y_true == -1) & (y_pred == 1)))
    TN = int(np.sum((y_true == -1) & (y_pred == -1)))
    FN = int(np.sum((y_true == 1)  & (y_pred == -1)))

    sens = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    spec = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    acc  = (TP + TN) / len(y_true)
    f1   = (2 * prec * sens / (prec + sens)) if (prec + sens) > 0 else 0.0

    # AUC approximé via le taux de vrais positifs et taux de faux positifs
    tpr = sens
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    auc_approx = (tpr + (1 - fpr)) / 2   # approximation de Wilcoxon

    return {
        "NFA": FP, "NFR": FN, "TP": TP, "TN": TN,
        "Sensibilité": round(sens, 3),
        "Spécificité": round(spec, 3),
        "Précision":   round(prec, 3),
        "F1":          round(f1, 3),
        "Exactitude":  round(acc, 3),
        "AUC":         round(auc_approx, 3),
    }


def train_svm_baseline(X_train, y_train, X_test, y_test):
    """SVM mono-critère : GridSearchCV sur C et gamma, score AUC."""
    param_grid = {"C": [0.1, 1, 10, 100, 500],
                  "gamma": [0.001, 0.01, 0.1, 1.0]}
    gs = GridSearchCV(
        SVC(kernel="rbf", random_state=0, max_iter=5000),
        param_grid, scoring="roc_auc", cv=3, n_jobs=-1
    )
    gs.fit(X_train, y_train)
    y_pred = gs.best_estimator_.predict(X_test)
    result = _metrics(y_test, y_pred)
    result["Méthode"] = "SVM GridSearch (baseline)"
    result["Params"]  = gs.best_params_
    return result


def train_random_forest(X_train, y_train, X_test, y_test):
    """Random Forest avec class_weight='balanced'."""
    param_grid = {"n_estimators": [100, 200],
                  "max_depth":    [None, 10, 20]}
    gs = GridSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=0),
        param_grid, scoring="roc_auc", cv=3, n_jobs=-1
    )
    gs.fit(X_train, y_train)
    y_pred = gs.best_estimator_.predict(X_test)
    result = _metrics(y_test, y_pred)
    result["Méthode"] = "Random Forest (balanced)"
    result["Params"]  = gs.best_params_
    return result


def train_gradient_boosting(X_train, y_train, X_test, y_test):
    """
    Gradient Boosting (sklearn) — équivalent fonctionnel de XGBoost.
    Le déséquilibre est compensé via sample_weight.
    """
    # Calcul du scale_pos_weight équivalent
    n_neg = np.sum(y_train == -1)
    n_pos = np.sum(y_train == 1)
    scale = n_neg / n_pos if n_pos > 0 else 1.0

    # Poids par échantillon : classe positive reçoit le poids "scale"
    sample_weight = np.where(y_train == 1, scale, 1.0)

    param_grid = {"n_estimators": [100, 200],
                  "max_depth":    [3, 5],
                  "learning_rate": [0.05, 0.1]}
    gs = GridSearchCV(
        GradientBoostingClassifier(random_state=0),
        param_grid, scoring="roc_auc", cv=3, n_jobs=-1
    )
    gs.fit(X_train, y_train, sample_weight=sample_weight)
    y_pred = gs.best_estimator_.predict(X_test)
    result = _metrics(y_test, y_pred)
    result["Méthode"] = "Gradient Boosting (scale_pos_weight)"
    result["Params"]  = gs.best_params_
    return result


def train_logistic_regression(X_train, y_train, X_test, y_test):
    """Régression Logistique avec class_weight='balanced'."""
    param_grid = {"C": [0.01, 0.1, 1, 10, 100]}
    gs = GridSearchCV(
        LogisticRegression(class_weight="balanced", max_iter=1000,
                           random_state=0),
        param_grid, scoring="roc_auc", cv=3, n_jobs=-1
    )
    gs.fit(X_train, y_train)
    y_pred = gs.best_estimator_.predict(X_test)
    result = _metrics(y_test, y_pred)
    result["Méthode"] = "Régression Logistique (balanced)"
    result["Params"]  = gs.best_params_
    return result


def run_all_competitors(X_train, y_train, X_test, y_test):
    """Lance les 4 méthodes concurrentes et retourne leurs résultats."""
    results = []
    for fn in [train_logistic_regression, train_svm_baseline,
               train_random_forest, train_gradient_boosting]:
        results.append(fn(X_train, y_train, X_test, y_test))
    return results
