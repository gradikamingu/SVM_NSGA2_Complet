"""
Visualisations spécifiques aux expérimentations multi-datasets.
Produit les figures correspondant aux tableaux du chapitre 4.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


COLORS = {
    "primary":   "#1F4E79",
    "secondary": "#2E75B6",
    "accent":    "#E74C3C",
    "green":     "#27AE60",
    "orange":    "#E67E22",
    "purple":    "#8E44AD",
    "gray":      "#7F8C8D",
    "light":     "#EBF5FB",
}

METHOD_COLORS = {
    "Régression Logistique (balanced)":          COLORS["gray"],
    "SVM GridSearch (baseline)":                 COLORS["orange"],
    "Random Forest (balanced)":                  COLORS["green"],
    "Gradient Boosting (scale_pos_weight)":      COLORS["purple"],
    "NSGA-II-SVM — Max Spécificité":             COLORS["secondary"],
    "NSGA-II-SVM — Compromis F1":                COLORS["primary"],
    "NSGA-II-SVM — Max Sensibilité":             COLORS["accent"],
}


def plot_pareto_multi(datasets_results, save_path=None):
    """
    Figure 4.1 — Front de Pareto sur les 4 datasets (subplots 2×2).
    datasets_results : liste de dict {name, pareto_front, baseline}
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Fronts de Pareto NSGA-II-SVM sur les 4 datasets\n"
                 "Optimisation bi-critère : NFA vs NFR",
                 fontsize=14, fontweight="bold", y=0.98)

    for ax, ds in zip(axes.flat, datasets_results):
        pareto = ds["pareto_front"]
        pts = sorted(pareto, key=lambda x: x["objectives"][0])
        nfa = [p["objectives"][0] for p in pts]
        nfr = [p["objectives"][1] for p in pts]

        ax.plot(nfa, nfr, "o-", color=COLORS["primary"],
                linewidth=1.5, markersize=5, label="Front de Pareto")
        ax.fill_between(nfa, nfr, max(nfr) * 1.1,
                        alpha=0.07, color=COLORS["primary"])

        # Baseline
        bsl = ds["baseline"]
        ax.scatter(bsl["NFA"], bsl["NFR"], color=COLORS["accent"],
                   s=120, zorder=5, marker="*", label="SVM mono-critère")

        ax.set_xlabel("NFA (Faux Positifs)", fontsize=9)
        ax.set_ylabel("NFR (Faux Négatifs)", fontsize=9)
        ax.set_title(ds["name"], fontweight="bold", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✔ Sauvegardé : {save_path}")
    plt.close()
    return fig


def plot_comparison_bars(comparison_data, dataset_name, save_path=None):
    """
    Figure 4.2 — Barres comparatives Sensibilité / Spécificité / F1
    pour toutes les méthodes sur un dataset donné.
    """
    methods  = [d["Méthode"] for d in comparison_data]
    sens     = [d["Sensibilité"] for d in comparison_data]
    spec     = [d["Spécificité"] for d in comparison_data]
    f1       = [d["F1"] for d in comparison_data]

    x   = np.arange(len(methods))
    w   = 0.25
    fig, ax = plt.subplots(figsize=(14, 6))

    bars_s = ax.bar(x - w, sens, w, label="Sensibilité", color=COLORS["accent"],   alpha=0.85)
    bars_p = ax.bar(x,     spec, w, label="Spécificité", color=COLORS["secondary"], alpha=0.85)
    bars_f = ax.bar(x + w, f1,   w, label="F1",          color=COLORS["primary"],   alpha=0.85)

    for bars in [bars_s, bars_p, bars_f]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7, rotation=45)

    short_labels = [m.replace("Gradient Boosting (scale_pos_weight)", "Grad. Boosting")
                     .replace("Régression Logistique (balanced)", "Rég. Logistique")
                     .replace("Random Forest (balanced)", "Random Forest")
                     .replace("SVM GridSearch (baseline)", "SVM baseline")
                     .replace("NSGA-II-SVM — ", "NSGA-II ") for m in methods]

    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(f"Comparaison des méthodes — {dataset_name}\n"
                 "Sensibilité · Spécificité · F1",
                 fontweight="bold", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.axvline(x=3.5, color="gray", linestyle=":", linewidth=1)
    ax.text(3.6, 1.08, "← Concurrents  |  NSGA-II →",
            fontsize=8, color="gray", va="top")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✔ Sauvegardé : {save_path}")
    plt.close()


def plot_sensitivity_analysis(sensitivity_results, save_path=None):
    """
    Figure 4.3 — Analyse de sensibilité aux paramètres NSGA-II.
    Barres : taille du front / convergence / SVM entraînés.
    """
    labels = [r["Configuration"] for r in sensitivity_results]
    sizes  = [r["Taille front"]  for r in sensitivity_results]
    svms   = [r["SVM entraînés"] for r in sensitivity_results]
    convs  = [r["Convergence (gen)"] for r in sensitivity_results]

    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Analyse de sensibilité aux paramètres NSGA-II\n"
                 "Dataset AVC — impact de N (population) et M (générations)",
                 fontweight="bold", fontsize=13)

    colors = [COLORS["accent"] if "référence" in l else COLORS["secondary"]
              for l in labels]

    for ax, vals, title, ylabel in zip(
        axes,
        [sizes, svms, convs],
        ["Taille du front de Pareto", "SVM entraînés (cache)", "Génération de convergence"],
        ["Nombre de solutions", "Nombre de SVM", "Génération"]
    ):
        bars = ax.bar(x, vals, color=colors, alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.01,
                    str(v), ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        ax.set_ylim(0, max(vals) * 1.2)

    ref_patch  = mpatches.Patch(color=COLORS["accent"],    label="Configuration de référence")
    var_patch  = mpatches.Patch(color=COLORS["secondary"], label="Variante")
    fig.legend(handles=[ref_patch, var_patch], loc="lower center",
               ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✔ Sauvegardé : {save_path}")
    plt.close()


def plot_global_summary(all_comparisons, save_path=None):
    """
    Figure 4.4 — Récapitulatif global : F1 de toutes les méthodes sur tous les datasets.
    Heatmap méthodes × datasets.
    """
    dataset_names = [c["name"] for c in all_comparisons]
    # Toutes les méthodes présentes
    all_methods = []
    for c in all_comparisons:
        for r in c["results"]:
            m = r["Méthode"]
            if m not in all_methods:
                all_methods.append(m)

    # Matrice F1
    matrix = np.zeros((len(all_methods), len(dataset_names)))
    for j, c in enumerate(all_comparisons):
        for r in c["results"]:
            i = all_methods.index(r["Méthode"])
            matrix[i, j] = r["F1"]

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0.5, vmax=1.0)

    ax.set_xticks(range(len(dataset_names)))
    ax.set_xticklabels(dataset_names, fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(all_methods)))
    short_m = [m.replace("NSGA-II-SVM — ", "NSGA-II ") for m in all_methods]
    ax.set_yticklabels(short_m, fontsize=9)

    for i in range(len(all_methods)):
        for j in range(len(dataset_names)):
            v = matrix[i, j]
            if v > 0:
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=9, fontweight="bold",
                        color="white" if v > 0.85 else "black")

    plt.colorbar(im, ax=ax, label="Score F1")
    ax.set_title("Score F1 par méthode et par dataset\n"
                 "Récapitulatif global des expérimentations",
                 fontweight="bold", fontsize=13)

    # Ligne de séparation entre méthodes concurrentes et NSGA-II
    n_competitors = sum(1 for m in all_methods if "NSGA-II" not in m)
    ax.axhline(n_competitors - 0.5, color="white", linewidth=2.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✔ Sauvegardé : {save_path}")
    plt.close()
