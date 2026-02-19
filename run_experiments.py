"""
Pipeline d'expérimentations — 3 datasets médicaux
AVC × Cancer du sein × Diabète
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os, time
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
    plot_sensitivity_analysis, plot_global_summary)

OUT = "results_experiments"
os.makedirs(OUT, exist_ok=True)
NSGA2_CFG = {"pop_size": 40, "n_generations": 60, "random_state": 42}


def _pareto_notable(pareto, X_train, y_train, X_test, y_test):
    ev = SVMEvaluator(X_train, y_train, X_test, y_test)
    metrics = [ev.full_metrics(ind["genes"]) for ind in pareto]
    return {
        "NSGA-II-SVM — Max Spécificité": max(metrics, key=lambda m: m["Spécificité"]),
        "NSGA-II-SVM — Compromis F1":    max(metrics, key=lambda m: m["Score F1"]),
        "NSGA-II-SVM — Max Sensibilité": max(metrics, key=lambda m: m["Sensibilité (Rappel)"]),
    }


def run_dataset(splits, info):
    X_train, X_val, X_test, y_train, y_val, y_test = splits
    bounds = [(1.0, info["c_plus_max"]), (1.0, 500.0), (1e-4, 1.0)]

    print(f"\n  {'─'*58}")
    print(f"  Dataset : {info['name']}  "
          f"(N={info['n']}, p={info['pct_pos']}%, features={info['n_features']})")
    print(f"  {'─'*58}")

    # NSGA-II
    print("  [NSGA-II] Optimisation...")
    t0 = time.time()
    ev = SVMEvaluator(X_train, y_train, X_val, y_val)
    pareto, history = nsga2(ev.evaluate, bounds, verbose=True, **NSGA2_CFG)
    print(f"  [NSGA-II] {time.time()-t0:.1f}s — {len(pareto)} sols — {len(ev._cache)} SVM")

    notable = _pareto_notable(pareto, X_train, y_train, X_test, y_test)

    # Concurrents
    print("  [Concurrents] Entraînement...")
    competitors = run_all_competitors(X_train, y_train, X_test, y_test)
    for c in competitors:
        print(f"    {c['Méthode']:<42} F1={c['F1']:.3f}  "
              f"Sens={c['Sensibilité']:.3f}  Spec={c['Spécificité']:.3f}")

    print("  [NSGA-II] Solutions remarquables :")
    nsga_rows = []
    for label, m in notable.items():
        row = {"Méthode": label, "NFA": m["NFA"], "NFR": m["NFR"],
               "Sensibilité": m["Sensibilité (Rappel)"],
               "Spécificité": m["Spécificité"], "Précision": m["Précision"],
               "F1": m["Score F1"], "Exactitude": m["Exactitude"],
               "AUC": round((m["Sensibilité (Rappel)"]+m["Spécificité"])/2, 3),
               "Params": f"C+={m['C+']:.1f},C-={m['C-']:.1f},γ={m['gamma']:.5f}"}
        nsga_rows.append(row)
        print(f"    {label:<42} F1={row['F1']:.3f}  "
              f"Sens={row['Sensibilité']:.3f}  Spec={row['Spécificité']:.3f}")

    baseline = next(r for r in competitors if r["Méthode"] == "SVM GridSearch (baseline)")
    all_results = competitors + nsga_rows
    for r in all_results: r["Dataset"] = info["name"]

    return {"name": info["name"], "pareto_front": pareto, "history": history,
            "baseline": baseline, "results": all_results}


if __name__ == "__main__":
    t0 = time.time()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  EXPÉRIMENTATIONS — AVC · Cancer du sein · Diabète          ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    print("\n[1/4] Chargement des datasets...")
    all_ds = load_all(42)
    for sp, info in all_ds:
        print(f"  • {info['name']:<20} N={info['n']:>5}  "
              f"positifs={info['pct_pos']}%  source={info['source'][:50]}")

    print("\n[2/4] NSGA-II + Méthodes concurrentes...")
    results = [run_dataset(sp, info) for sp, info in all_ds]

    print("\n[3/4] Analyse de sensibilité (dataset AVC)...")
    avc_splits = all_ds[0][0]
    sens_res = run_sensitivity(*avc_splits[:4])

    print("\n[4/4] Visualisations...")
    plot_pareto_multi(results, save_path=os.path.join(OUT,"fig4_1_pareto_multi.png"))
    for ds_res in results:
        safe = ds_res["name"].replace(" ","_")
        plot_comparison_bars(ds_res["results"], ds_res["name"],
                             save_path=os.path.join(OUT,f"fig4_2_comparaison_{safe}.png"))
    plot_sensitivity_analysis(sens_res, save_path=os.path.join(OUT,"fig4_3_sensibilite.png"))
    plot_global_summary(results, save_path=os.path.join(OUT,"fig4_4_resume_global.png"))

    # CSV
    all_rows = [r for ds in results for r in ds["results"]]
    pd.DataFrame(all_rows).to_csv(os.path.join(OUT,"resultats_complets.csv"),
                                   index=False, encoding="utf-8-sig")
    pd.DataFrame(sens_res).to_csv(os.path.join(OUT,"sensibilite.csv"),
                                   index=False, encoding="utf-8-sig")

    print(f"\n✅  Terminé en {(time.time()-t0)/60:.1f} min")
    print(f"   Résultats dans : {OUT}/")
    for f in sorted(os.listdir(OUT)): print(f"     • {f}")
