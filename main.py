"""
╔══════════════════════════════════════════════════════════════════════════╗
║   Optimisation bi-critère des hyper-paramètres SVM par NSGA-II          ║
║   Application : Évaluation binaire du risque d'AVC                      ║
╚══════════════════════════════════════════════════════════════════════════╝

Pipeline complet :
    1. Génération / chargement du dataset AVC
    2. Prétraitement et split train / validation / test
    3. Optimisation NSGA-II sur le jeu de validation → Front de Pareto
    4. Comparaison avec un SVM mono-critère (GridSearch + AUC)
    5. Visualisations et rapport
"""

import sys
import warnings
warnings.filterwarnings("ignore")
import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

# Ajouter le dossier src au path
sys.path.insert(0, os.path.dirname(__file__))

from src.nsga2 import nsga2
from src.svm_evaluator import SVMEvaluator
from src.visualisation import (plot_pareto_front, plot_hyperparameters,
                                plot_convergence, plot_confusion_matrix,
                                plot_roc_comparison)
from data.generate_dataset import generate_stroke_dataset


# ══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════

NSGA2_CONFIG = {
    'pop_size':      40,      # Taille de la population N
    'n_generations': 60,      # Nombre de générations M
    'random_state':  42,
}

# Bornes des hyper-paramètres : (min, max)
BOUNDS = [
    (1.0,   5000.0),    # C+ (pénalité classe positive / AVC)
    (1.0,   500.0),     # C- (pénalité classe négative / non-AVC)
    (1e-4,  1.0),       # gamma (noyau RBF)
]

RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════
#  1. DONNÉES
# ══════════════════════════════════════════════════════════════════════════

def load_data():
    print("\n[1/5] Chargement du dataset AVC...")
    df = generate_stroke_dataset(n_samples=1000, positive_rate=0.15, random_state=42)

    X = df.drop(columns=['stroke']).values
    y = df['stroke'].values  # {+1, -1}

    print(f"      {len(df)} individus | "
          f"{(y==1).sum()} AVC ({(y==1).mean():.1%}) | "
          f"{(y==-1).sum()} non-AVC ({(y==-1).mean():.1%})")

    # Split : 70% train / 15% validation / 15% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=42)
    # 0.176 ≈ 15/85 pour obtenir 15% du total

    print(f"      Train : {len(X_train)} | Val : {len(X_val)} | Test : {len(X_test)}")

    # Normalisation (StandardScaler ajusté sur train uniquement)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


# ══════════════════════════════════════════════════════════════════════════
#  2. BASELINE : SVM MONO-CRITÈRE (GridSearch + AUC)
# ══════════════════════════════════════════════════════════════════════════

def train_baseline(X_train, y_train, X_val, y_val):
    print("\n[2/5] Entraînement du SVM mono-critère (GridSearch + AUC)...")

    param_grid = {
        'C':     [0.1, 1, 10, 100, 500],
        'gamma': [0.001, 0.01, 0.1, 1.0],
    }
    gs = GridSearchCV(
        SVC(kernel='rbf', random_state=0),
        param_grid, scoring='roc_auc', cv=3, n_jobs=-1, verbose=0
    )
    gs.fit(X_train, y_train)
    best = gs.best_estimator_

    y_pred_val = best.predict(X_val)
    NFA = int(np.sum((y_val == -1) & (y_pred_val == 1)))
    NFR = int(np.sum((y_val == 1)  & (y_pred_val == -1)))
    TP  = int(np.sum((y_val == 1)  & (y_pred_val == 1)))
    TN  = int(np.sum((y_val == -1) & (y_pred_val == -1)))

    sens = TP / (TP + NFR) if (TP + NFR) > 0 else 0
    spec = TN / (TN + NFA) if (TN + NFA) > 0 else 0

    print(f"      Meilleurs params : C={gs.best_params_['C']}, γ={gs.best_params_['gamma']}")
    print(f"      NFA={NFA} | NFR={NFR} | Sensibilité={sens:.3f} | Spécificité={spec:.3f}")

    return {
        'model': best,
        'NFA': NFA, 'NFR': NFR, 'TP': TP, 'TN': TN,
        'Sensibilité (Rappel)': round(sens, 4),
        'Spécificité': round(spec, 4),
        'C': gs.best_params_['C'],
        'gamma': gs.best_params_['gamma'],
    }


# ══════════════════════════════════════════════════════════════════════════
#  3. OPTIMISATION NSGA-II
# ══════════════════════════════════════════════════════════════════════════

def run_nsga2(X_train, y_train, X_val, y_val):
    print("\n[3/5] Optimisation NSGA-II bi-critère (NFA, NFR)...")

    evaluator = SVMEvaluator(X_train, y_train, X_val, y_val)
    evaluate_fn = evaluator.evaluate

    t0 = time.time()
    pareto_front, history = nsga2(
        evaluate_fn=evaluate_fn,
        bounds=BOUNDS,
        **NSGA2_CONFIG,
        verbose=True,
    )
    elapsed = time.time() - t0
    print(f"\n      Temps d'exécution : {elapsed:.1f}s")
    print(f"      Nombre de SVM entraînés (avec cache) : {len(evaluator._cache)}")

    return pareto_front, history, evaluator


# ══════════════════════════════════════════════════════════════════════════
#  4. ANALYSE DU FRONT DE PARETO
# ══════════════════════════════════════════════════════════════════════════

def analyse_pareto(pareto_front, evaluator):
    print("\n[4/5] Analyse du Front de Pareto...")

    # Calcul des métriques complètes pour chaque solution
    solutions = []
    for ind in pareto_front:
        m = evaluator.full_metrics(ind['genes'])
        solutions.append(m)

    df = pd.DataFrame(solutions)
    df = df.sort_values('NFA')

    print("\n  Solutions du Front de Pareto (triées par NFA croissant) :")
    cols = ['C+', 'C-', 'gamma', 'NFA', 'NFR',
            'Sensibilité (Rappel)', 'Spécificité', 'Score F1', 'Exactitude']
    print(df[cols].to_string(index=False))

    # Sauvegarde CSV
    csv_path = os.path.join(RESULTS_DIR, 'pareto_front.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n  ✔ Front de Pareto sauvegardé : {csv_path}")

    # Identifier 3 solutions remarquables
    idx_best_sens = df['Sensibilité (Rappel)'].idxmax()
    idx_best_spec = df['Spécificité'].idxmax()
    idx_best_f1   = df['Score F1'].idxmax()

    print("\n  ─── Solutions remarquables ───────────────────────────────")
    for label, idx in [('Max Sensibilité (min NFR)', idx_best_sens),
                        ('Max Spécificité (min NFA)', idx_best_spec),
                        ('Max Score F1 (compromis)', idx_best_f1)]:
        row = df.loc[idx]
        print(f"  [{label}]")
        print(f"    C+={row['C+']:.1f} | C-={row['C-']:.1f} | γ={row['gamma']:.5f}")
        print(f"    NFA={row['NFA']} | NFR={row['NFR']} | F1={row['Score F1']:.3f}")
    print("  ──────────────────────────────────────────────────────────")

    return df, df.loc[idx_best_f1]


# ══════════════════════════════════════════════════════════════════════════
#  5. VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════

def make_visualisations(pareto_front, history, evaluator, baseline, best_solution, df_pareto):
    print("\n[5/5] Génération des visualisations...")

    plot_pareto_front(
        pareto_front, baseline_metrics=baseline,
        save_path=os.path.join(RESULTS_DIR, '1_pareto_front.png')
    )
    plot_hyperparameters(
        pareto_front,
        save_path=os.path.join(RESULTS_DIR, '2_hyperparametres.png')
    )
    plot_convergence(
        history,
        save_path=os.path.join(RESULTS_DIR, '3_convergence.png')
    )
    plot_confusion_matrix(
        best_solution.to_dict(), title='Meilleur compromis (F1 max)',
        save_path=os.path.join(RESULTS_DIR, '4_matrice_confusion.png')
    )
    plot_roc_comparison(
        pareto_front, baseline_metrics=baseline, evaluator=evaluator,
        save_path=os.path.join(RESULTS_DIR, '5_roc_comparison.png')
    )


# ══════════════════════════════════════════════════════════════════════════
#  POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   SVM × NSGA-II — Prédiction du risque d'AVC            ║")
    print("╚══════════════════════════════════════════════════════════╝")

    X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_data()
    baseline    = train_baseline(X_train, y_train, X_val, y_val)
    pareto_front, history, evaluator = run_nsga2(X_train, y_train, X_val, y_val)
    df_pareto, best_sol = analyse_pareto(pareto_front, evaluator)
    make_visualisations(pareto_front, history, evaluator, baseline, best_sol, df_pareto)

    print("\n✅  Pipeline terminé. Résultats dans le dossier 'results/'.")
    print("   Fichiers générés :")
    for f in sorted(os.listdir(RESULTS_DIR)):
        print(f"     • {f}")
