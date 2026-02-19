"""
╔══════════════════════════════════════════════════════════════════════════╗
║   Pipeline d'expérimentations complet — Chapitre 4                      ║
║   Correspond exactement aux sections 4.2 à 4.6 du document              ║
╚══════════════════════════════════════════════════════════════════════════╝

Exécution : python run_experiments.py
Durée estimée : 8 à 15 minutes selon le hardware.

Résultats produits dans results_experiments/ :
    - pareto_multi.png           (Fig. 4.1 — Fronts de Pareto × 4 datasets)
    - comparaison_AVC.png        (Fig. 4.2a)
    - comparaison_BreastCancer.png
    - comparaison_Diabetes.png
    - comparaison_Fraud.png
    - sensibilite_nsga2.png      (Fig. 4.3 — Analyse de sensibilité)
    - resume_global.png          (Fig. 4.4 — Heatmap F1 globale)
    - resultats_complets.csv     (toutes les métriques)
    - rapport_console.txt        (sortie texte complète)
"""

import warnings
warnings.filterwarnings("ignore")

import sys, os, time, io
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

from src.nsga2 import nsga2
from src.svm_evaluator import SVMEvaluator
from experiments.datasets import load_all
from experiments.competitors import run_all_competitors
from experiments.sensitivity_analysis import run_sensitivity
from experiments.visualisation_experiments import (
    plot_pareto_multi, plot_comparison_bars,
    plot_sensitivity_analysis, plot_global_summary
)

OUT = "results_experiments"
os.makedirs(OUT, exist_ok=True)

BOUNDS_DEFAULT = [(1.0, 5000.0), (1.0, 500.0), (1e-4, 1.0)]
NSGA2_CFG = {"pop_size": 40, "n_generations": 60, "random_state": 42}


# ── Helpers ────────────────────────────────────────────────────────────────

def _pareto_notable(pareto, evaluator):
    """Extrait les 3 solutions remarquables du front de Pareto."""
    from src.svm_evaluator import SVMEvaluator
    metrics_list = [evaluator.full_metrics(ind["genes"]) for ind in pareto]

    best_f1   = max(metrics_list, key=lambda m: m["Score F1"])
    best_sens = max(metrics_list, key=lambda m: m["Sensibilité (Rappel)"])
    best_spec = max(metrics_list, key=lambda m: m["Spécificité"])

    return {
        "NSGA-II-SVM — Max Spécificité": best_spec,
        "NSGA-II-SVM — Compromis F1":   best_f1,
        "NSGA-II-SVM — Max Sensibilité": best_sens,
    }


def _nsga2_row(label, m):
    return {
        "Méthode":      label,
        "NFA":          m["NFA"],
        "NFR":          m["NFR"],
        "Sensibilité":  m["Sensibilité (Rappel)"],
        "Spécificité":  m["Spécificité"],
        "Précision":    m["Précision"],
        "F1":           m["Score F1"],
        "Exactitude":   m["Exactitude"],
        "AUC":          round((m["Sensibilité (Rappel)"] + m["Spécificité"]) / 2, 3),
        "Params":       f"C+={m['C+']:.1f}, C-={m['C-']:.1f}, γ={m['gamma']:.5f}",
    }


# ── SECTION 4.2-4.4 : NSGA-II + concurrents sur chaque dataset ────────────

def run_dataset(splits, info):
    X_train, X_val, X_test, y_train, y_val, y_test = splits
    bounds = [(1.0, info["c_plus_max"]), (1.0, 500.0), (1e-4, 1.0)]

    print(f"\n  {'─'*55}")
    print(f"  Dataset : {info['name']}  "
          f"(N={info['n']}, features={info['n_features']}, "
          f"positifs={info['pct_pos']}%)")
    print(f"  {'─'*55}")

    # ── NSGA-II ──────────────────────────────────────────────────
    print(f"  [NSGA-II] Optimisation bi-critère...")
    t0 = time.time()
    evaluator = SVMEvaluator(X_train, y_train, X_val, y_val)
    pareto, history = nsga2(
        evaluate_fn=evaluator.evaluate,
        bounds=bounds,
        verbose=True,
        **NSGA2_CFG,
    )
    elapsed = time.time() - t0
    print(f"  [NSGA-II] Terminé en {elapsed:.1f}s — "
          f"{len(pareto)} solutions — {len(evaluator._cache)} SVM entraînés")

    # Métriques sur jeu de TEST pour les 3 solutions remarquables
    # On ré-évalue sur X_test / y_test
    evaluator_test = SVMEvaluator(X_train, y_train, X_test, y_test)
    notable = _pareto_notable(pareto, evaluator_test)

    # ── Concurrents ───────────────────────────────────────────────
    print(f"  [Concurrents] Entraînement des 4 méthodes...")
    competitors = run_all_competitors(X_train, y_train, X_test, y_test)
    for c in competitors:
        print(f"    {c['Méthode']:<40} F1={c['F1']:.3f}  "
              f"Sens={c['Sensibilité']:.3f}  Spec={c['Spécificité']:.3f}")

    # ── Résumé NSGA-II ─────────────────────────────────────────────
    print(f"  [NSGA-II] Solutions remarquables (test) :")
    nsga_rows = []
    for label, m in notable.items():
        row = _nsga2_row(label, m)
        nsga_rows.append(row)
        print(f"    {label:<40} F1={row['F1']:.3f}  "
              f"Sens={row['Sensibilité']:.3f}  Spec={row['Spécificité']:.3f}")

    all_results = competitors + nsga_rows
    for r in all_results:
        r["Dataset"] = info["name"]

    # Baseline pour le graphe du front de Pareto
    baseline_metrics = next(r for r in competitors
                            if r["Méthode"] == "SVM GridSearch (baseline)")

    return {
        "name":         info["name"],
        "pareto_front": pareto,
        "history":      history,
        "evaluator":    evaluator,
        "baseline":     baseline_metrics,
        "results":      all_results,
    }


# ── SECTION 4.6 : Analyse de sensibilité ──────────────────────────────────

def run_sensitivity_section(avc_splits):
    X_train, X_val, X_test, y_train, y_val, y_test = avc_splits
    print(f"\n{'═'*60}")
    print("  Section 4.6 — Analyse de sensibilité aux paramètres NSGA-II")
    print(f"{'═'*60}")
    return run_sensitivity(X_train, y_train, X_val, y_val)


# ── MAIN ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t_global = time.time()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║   EXPÉRIMENTATIONS — Chapitre 4                          ║")
    print("║   SVM × NSGA-II — 4 datasets × 4 méthodes               ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # ── Chargement des datasets ──────────────────────────────────
    print("\n[1/4] Chargement des datasets...")
    all_datasets = load_all(random_state=42)
    for splits, info in all_datasets:
        print(f"  • {info['name']:<20} N={info['n']:>6}  "
              f"features={info['n_features']:>2}  positifs={info['pct_pos']}%")

    # ── Expérimentations par dataset ─────────────────────────────
    print("\n[2/4] Expérimentations par dataset (NSGA-II + concurrents)...")
    all_experiment_results = []
    for splits, info in all_datasets:
        ds_result = run_dataset(splits, info)
        all_experiment_results.append(ds_result)

    # ── Analyse de sensibilité (sur AVC uniquement) ───────────────
    print("\n[3/4] Analyse de sensibilité...")
    avc_splits = all_datasets[0][0]
    sensitivity_results = run_sensitivity_section(avc_splits)

    # ── Visualisations ────────────────────────────────────────────
    print(f"\n[4/4] Génération des visualisations...")

    # Fig 4.1 — Fronts de Pareto × 4 datasets
    plot_pareto_multi(
        all_experiment_results,
        save_path=os.path.join(OUT, "fig4_1_pareto_multi.png")
    )

    # Fig 4.2 — Barres comparatives par dataset
    for ds_res in all_experiment_results:
        safe_name = ds_res["name"].replace(" ", "_")
        plot_comparison_bars(
            ds_res["results"],
            dataset_name=ds_res["name"],
            save_path=os.path.join(OUT, f"fig4_2_comparaison_{safe_name}.png")
        )

    # Fig 4.3 — Analyse de sensibilité
    plot_sensitivity_analysis(
        sensitivity_results,
        save_path=os.path.join(OUT, "fig4_3_sensibilite_nsga2.png")
    )

    # Fig 4.4 — Heatmap globale F1
    plot_global_summary(
        all_experiment_results,
        save_path=os.path.join(OUT, "fig4_4_resume_global.png")
    )

    # ── Export CSV ─────────────────────────────────────────────────
    all_rows = []
    for ds_res in all_experiment_results:
        all_rows.extend(ds_res["results"])

    df_results = pd.DataFrame(all_rows)
    cols_order = ["Dataset", "Méthode", "NFA", "NFR",
                  "Sensibilité", "Spécificité", "Précision", "F1",
                  "Exactitude", "AUC", "Params"]
    df_results = df_results[[c for c in cols_order if c in df_results.columns]]
    csv_path = os.path.join(OUT, "resultats_complets.csv")
    df_results.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"  ✔ CSV exporté : {csv_path}")

    df_sens = pd.DataFrame(sensitivity_results)
    df_sens.to_csv(os.path.join(OUT, "sensibilite_nsga2.csv"),
                   index=False, encoding="utf-8-sig")

    # ── Affichage final ─────────────────────────────────────────────
    total = time.time() - t_global
    print(f"\n{'═'*60}")
    print(f"  ✅  Toutes les expérimentations terminées en {total/60:.1f} min")
    print(f"  Résultats dans : {OUT}/")
    for f in sorted(os.listdir(OUT)):
        print(f"    • {f}")
    print(f"{'═'*60}")

    # ── Résumé console des tableaux du document ─────────────────────
    print("\n  ── TABLEAU 4.4 (extrait) ─────────────────────────────────")
    pivot = df_results[df_results["Méthode"].isin([
        "SVM GridSearch (baseline)",
        "NSGA-II-SVM — Compromis F1",
    ])][["Dataset", "Méthode", "Sensibilité", "Spécificité", "F1", "AUC"]]
    print(pivot.to_string(index=False))

    print("\n  ── TABLEAU 4.6 (sensibilité) ─────────────────────────────")
    print(df_sens[["Configuration", "Taille front", "NFA_min",
                   "NFR_min", "SVM entraînés", "Convergence (gen)"]].to_string(index=False))
