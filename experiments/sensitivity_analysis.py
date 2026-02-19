"""
Analyse de sensibilité aux paramètres NSGA-II (Section 4.6 du chapitre).

Variantes testées :
    - N = 20  (population réduite)
    - N = 40  (référence)
    - N = 80  (population élargie)
    - M = 30  (arrêt précoce)
    - M = 60  (référence)
"""

import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.nsga2 import nsga2
from src.svm_evaluator import SVMEvaluator


BOUNDS = [
    (1.0,   5000.0),
    (1.0,   500.0),
    (1e-4,  1.0),
]


def run_sensitivity(X_train, y_train, X_val, y_val):
    """
    Teste 5 configurations NSGA-II et retourne les résultats.
    """
    configs = [
        {"label": "N=20, M=60",  "pop_size": 20, "n_generations": 60},
        {"label": "N=40, M=60 (référence)", "pop_size": 40, "n_generations": 60},
        {"label": "N=80, M=60",  "pop_size": 80, "n_generations": 60},
        {"label": "N=40, M=30",  "pop_size": 40, "n_generations": 30},
        {"label": "N=40, M=20",  "pop_size": 40, "n_generations": 20},
    ]

    results = []
    for cfg in configs:
        evaluator = SVMEvaluator(X_train, y_train, X_val, y_val)
        pareto, history = nsga2(
            evaluate_fn=evaluator.evaluate,
            bounds=BOUNDS,
            pop_size=cfg["pop_size"],
            n_generations=cfg["n_generations"],
            random_state=42,
            verbose=False,
        )

        # Objectifs min sur le front final
        nfa_vals = [ind["objectives"][0] for ind in pareto]
        nfr_vals = [ind["objectives"][1] for ind in pareto]

        # Génération de convergence : première génération où NFA_min atteint son min final
        nfa_min_final = min(nfa_vals)
        nfr_min_final = min(nfr_vals)
        conv_gen = cfg["n_generations"]
        for h in history:
            if h["nfa_min"] == nfa_min_final and h["nfr_min"] == nfr_min_final:
                conv_gen = h["generation"]
                break

        results.append({
            "Configuration": cfg["label"],
            "N": cfg["pop_size"],
            "M": cfg["n_generations"],
            "Taille front": len(pareto),
            "NFA_min": nfa_min_final,
            "NFR_min": nfr_min_final,
            "SVM entraînés": len(evaluator._cache),
            "Convergence (gen)": conv_gen,
        })
        print(f"    [{cfg['label']}] → Front: {len(pareto)} sol. | "
              f"NFA_min={nfa_min_final} | NFR_min={nfr_min_final} | "
              f"Convergence: gen {conv_gen}")

    return results
