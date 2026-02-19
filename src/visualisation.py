"""
Visualisation des résultats de l'optimisation NSGA-II / SVM.

Graphiques produits :
    1. Front de Pareto NFA vs NFR
    2. Évolution du front de Pareto par génération
    3. Distribution des hyper-paramètres sur le front de Pareto
    4. Comparaison front ROC multi-objectif vs SVM mono-critère
    5. Matrice de confusion pour une solution choisie
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import os

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.facecolor': 'white',
})

COLORS = {
    'pareto':   '#E63946',
    'baseline': '#457B9D',
    'fill':     '#A8DADC',
    'grid':     '#F1FAEE',
    'accent':   '#F4A261',
    'dark':     '#1D3557',
}


def plot_pareto_front(pareto_front, baseline_metrics=None, save_path=None):
    """
    Trace le front de Pareto NFA vs NFR avec comparaison optionnelle
    à un SVM mono-critère.
    """
    nfa = [ind['objectives'][0] for ind in pareto_front]
    nfr = [ind['objectives'][1] for ind in pareto_front]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor(COLORS['grid'])
    ax.grid(True, linestyle='--', alpha=0.6, color='white', linewidth=1.2)

    # Remplissage sous le front
    ax.fill_between(nfa, nfr, alpha=0.15, color=COLORS['pareto'], label='Zone dominée')

    # Front de Pareto
    ax.plot(nfa, nfr, 'o-', color=COLORS['pareto'], linewidth=2.2,
            markersize=7, label='Front de Pareto (NSGA-II)', zorder=5)

    # SVM baseline (mono-critère)
    if baseline_metrics:
        ax.scatter(baseline_metrics['NFA'], baseline_metrics['NFR'],
                   marker='*', s=250, color=COLORS['baseline'], zorder=6,
                   label=f"SVM mono-critère (GridSearch)\n"
                         f"NFA={baseline_metrics['NFA']}, NFR={baseline_metrics['NFR']}")

    # Annotations extrêmes (sans flèches pour compatibilité matplotlib)
    if len(nfa) > 1:
        ax.text(nfa[-1] + 0.3, nfr[-1] + 1, 'Max sensibilité\n(NFR minimal)',
                fontsize=8, color=COLORS['dark'],
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
        ax.text(nfa[0] + 0.3, nfr[0] + 1, 'Max spécificité\n(NFA minimal)',
                fontsize=8, color=COLORS['dark'],
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

    ax.set_xlabel('NFA  (Faux Positifs : sains → AVC)', fontweight='bold')
    ax.set_ylabel('NFR  (Faux Négatifs : AVC → sains)', fontweight='bold')
    ax.set_title('Front de Pareto — Optimisation bi-critère SVM (NSGA-II)\n'
                 'Application : Évaluation binaire du risque d\'AVC', fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✔ Sauvegardé : {save_path}")
    return fig


def plot_hyperparameters(pareto_front, save_path=None):
    """
    Distribution des hyper-paramètres (C+, C-, γ) sur le front de Pareto.
    """
    nfa = [ind['objectives'][0] for ind in pareto_front]
    nfr = [ind['objectives'][1] for ind in pareto_front]
    Cp  = [ind['genes'][0] for ind in pareto_front]
    Cm  = [ind['genes'][1] for ind in pareto_front]
    gam = [ind['genes'][2] for ind in pareto_front]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    params = [
        (Cp,  'C⁺ (pénalité classe positive)',   COLORS['pareto']),
        (Cm,  'C⁻ (pénalité classe négative)',    COLORS['baseline']),
        (gam, 'γ  (paramètre du noyau RBF)',       COLORS['accent']),
    ]

    for ax, (values, label, color) in zip(axes, params):
        ax.set_facecolor(COLORS['grid'])
        ax.grid(True, linestyle='--', alpha=0.6, color='white')

        sc = ax.scatter(nfa, nfr, c=values, cmap='RdYlGn_r',
                        s=80, edgecolors='white', linewidth=0.8, zorder=5)
        ax.plot(nfa, nfr, '--', color='gray', alpha=0.4, zorder=3)

        cb = plt.colorbar(sc, ax=ax, shrink=0.85)
        cb.set_label(label, fontsize=8)
        ax.set_xlabel('NFA (Faux Positifs)', fontweight='bold')
        ax.set_ylabel('NFR (Faux Négatifs)', fontweight='bold')
        ax.set_title(label, fontweight='bold', pad=10)

    plt.suptitle('Distribution des hyper-paramètres sur le Front de Pareto',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✔ Sauvegardé : {save_path}")
    return fig


def plot_convergence(history, save_path=None):
    """
    Convergence de l'algorithme : taille du front et valeurs minimales.
    """
    generations = [h['generation'] for h in history]
    pareto_size = [h['pareto_size'] for h in history]
    nfa_min     = [h['nfa_min'] for h in history]
    nfr_min     = [h['nfr_min'] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Taille du front de Pareto
    axes[0].set_facecolor(COLORS['grid'])
    axes[0].grid(True, linestyle='--', alpha=0.6, color='white')
    axes[0].plot(generations, pareto_size, color=COLORS['pareto'],
                 linewidth=2, marker='o', markersize=3)
    axes[0].fill_between(generations, pareto_size, alpha=0.2, color=COLORS['pareto'])
    axes[0].set_xlabel('Génération', fontweight='bold')
    axes[0].set_ylabel('Nombre de solutions', fontweight='bold')
    axes[0].set_title('Taille du Front de Pareto par génération', fontweight='bold')

    # NFA_min et NFR_min
    axes[1].set_facecolor(COLORS['grid'])
    axes[1].grid(True, linestyle='--', alpha=0.6, color='white')
    axes[1].plot(generations, nfa_min, color=COLORS['baseline'],
                 linewidth=2, label='NFA min', marker='s', markersize=3)
    axes[1].plot(generations, nfr_min, color=COLORS['accent'],
                 linewidth=2, label='NFR min', marker='^', markersize=3)
    axes[1].set_xlabel('Génération', fontweight='bold')
    axes[1].set_ylabel('Nombre d\'erreurs (min)', fontweight='bold')
    axes[1].set_title('Convergence des objectifs (NFA_min, NFR_min)', fontweight='bold')
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✔ Sauvegardé : {save_path}")
    return fig


def plot_confusion_matrix(metrics, title='Solution choisie', save_path=None):
    """
    Matrice de confusion pour une solution donnée.
    """
    TP, FP = metrics['TP'], metrics['FP']
    FN, TN = metrics['FN'], metrics['TN']

    cm = np.array([[TP, FN], [FP, TN]])
    labels = [['TP\n(Vrai Positif)', 'FN\n(Faux Négatif)'],
              ['FP\n(Faux Positif)', 'TN\n(Vrai Négatif)']]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest',
                   cmap=plt.cm.Blues if TP + TN > FP + FN else plt.cm.Reds)

    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, f"{labels[i][j]}\n{cm[i, j]}",
                    ha='center', va='center', fontsize=12,
                    fontweight='bold', color=color)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Prédit Positif (AVC)', 'Prédit Négatif'], fontsize=10)
    ax.set_yticklabels(['Réel Positif (AVC)', 'Réel Négatif'], fontsize=10)
    ax.set_xlabel('Prédiction', fontweight='bold', labelpad=10)
    ax.set_ylabel('Réalité', fontweight='bold', labelpad=10)
    ax.set_title(f'Matrice de confusion — {title}\n'
                 f'C+={metrics["C+"]:.1f} | C-={metrics["C-"]:.1f} | γ={metrics["gamma"]:.4f}',
                 fontweight='bold')

    # Métriques sous la figure
    info = (f"Sensibilité={metrics['Sensibilité (Rappel)']:.3f} | "
            f"Spécificité={metrics['Spécificité']:.3f} | "
            f"F1={metrics['Score F1']:.3f} | "
            f"Exactitude={metrics['Exactitude']:.3f}")
    fig.text(0.5, -0.03, info, ha='center', fontsize=9, color=COLORS['dark'])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✔ Sauvegardé : {save_path}")
    return fig


def plot_roc_comparison(pareto_front, baseline_metrics, evaluator, save_path=None):
    """
    Compare le 'front ROC' multi-objectif à la courbe ROC d'un SVM mono-critère.
    Chaque point du front de Pareto est un modèle SVM distinct.
    """
    # Front de Pareto : taux FA et FR
    n_val = len(evaluator.y_val)
    n_pos_val = evaluator.n_pos
    n_neg_val = evaluator.n_neg

    fa_pareto = [ind['objectives'][0] / n_neg_val for ind in pareto_front]
    fr_pareto = [ind['objectives'][1] / n_pos_val for ind in pareto_front]

    # SVM baseline
    fa_base = baseline_metrics['NFA'] / n_neg_val
    fr_base = baseline_metrics['NFR'] / n_pos_val

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor(COLORS['grid'])
    ax.grid(True, linestyle='--', alpha=0.6, color='white')

    # Diagonale aléatoire
    ax.plot([0, 1], [1, 0], '--', color='gray', alpha=0.5, label='Classifieur aléatoire')

    # Front ROC multi-objectif
    ax.plot(fa_pareto, fr_pareto, 'o-', color=COLORS['pareto'],
            linewidth=2.5, markersize=7, label='Front ROC multi-objectif (NSGA-II)')
    ax.fill_between(fa_pareto, fr_pareto, alpha=0.15, color=COLORS['pareto'])

    # Baseline
    ax.scatter(fa_base, fr_base, marker='*', s=300, color=COLORS['baseline'], zorder=6,
               label=f'SVM mono-critère (AUC)')

    ax.set_xlabel('Taux de Faux Positifs (FA rate = NFA / N⁻)', fontweight='bold')
    ax.set_ylabel('Taux de Faux Négatifs (FR rate = NFR / N⁺)', fontweight='bold')
    ax.set_title('Front ROC multi-objectif vs SVM mono-critère\n'
                 'Chaque point = un SVM avec des hyper-paramètres distincts',
                 fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✔ Sauvegardé : {save_path}")
    return fig
