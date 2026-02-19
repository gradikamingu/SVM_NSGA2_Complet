# SVM × NSGA-II

Optimisation bi-critère des hyperparamètres SVM par l'algorithme NSGA-II,
appliquée au classement binaire.

## Structure du projet

```
svm_nsga2_avc/
├── main.py                          ← Pipeline principal (dataset AVC uniquement)
├── run_experiments.py               ← Expérimentations complètes (Chapitre 4)
├── requirements.txt
│
├── src/
│   ├── nsga2.py                     ← Algorithme NSGA-II from scratch
│   ├── svm_evaluator.py             ← Évaluation bi-critère (NFA, NFR)
│   └── visualisation.py             ← Visualisations (front de Pareto, etc.)
│
├── data/
│   └── generate_dataset.py          ← Génération du dataset AVC synthétique
│
└── experiments/
    ├── datasets.py                  ← Chargement des 4 datasets
    ├── competitors.py               ← Méthodes concurrentes (RF, GradBoost, LogReg)
    ├── sensitivity_analysis.py      ← Analyse de sensibilité N/M
    └── visualisation_experiments.py ← Figures du Chapitre 4
```

## Utilisation

### Pipeline simple (dataset AVC uniquement)
```bash
python main.py
```
Durée : ~1 minute. Résultats dans `results/`.

### Expérimentations complètes (Chapitre 4)
```bash
python run_experiments.py
```
Durée : ~8 à 15 minutes selon le hardware.
Résultats dans `results_experiments/` :

| Fichier | Contenu |
|---|---|
| `fig4_1_pareto_multi.png` | Fronts de Pareto × 4 datasets |
| `fig4_2_comparaison_*.png` | Barres comparatives par dataset |
| `fig4_3_sensibilite_nsga2.png` | Impact de N et M |
| `fig4_4_resume_global.png` | Heatmap F1 globale |
| `resultats_complets.csv` | Toutes les métriques |
| `sensibilite_nsga2.csv` | Résultats analyse de sensibilité |

## Datasets

| Dataset | N | Features | % Positifs | Source |
|---|---|---|---|---|
| AVC (principal) | 1 000 | 8 | 15,0 % | Synthétique |
| Breast Cancer | 569 | 30 | 37,3 % | sklearn (UCI) |
| Diabetes | 768 | 8 | 34,9 % | Synthétique (dist. Pima) |
| Fraud | 3 000 | 10 | 0,5 % | Synthétique (déséquilibre extrême) |

## Méthodes comparées

| Méthode | Section |
|---|---|
| SVM GridSearch + AUC (baseline) | 4.3 |
| Régression Logistique (balanced) | 4.5 |
| Random Forest (balanced) | 4.5 |
| Gradient Boosting (scale_pos_weight) | 4.5 |
| **NSGA-II-SVM (notre méthode)** | 4.3 – 4.4 |

## Installation

```bash
pip install numpy pandas scikit-learn matplotlib
```

## Paramètres NSGA-II

| Paramètre | Valeur |
|---|---|
| Population N | 40 |
| Générations M | 60 |
| Croisement | SBX (ηc = 20) |
| Mutation | Polynomiale (ηm = 20) |
| Bornes C+ | [1, 5 000] |
| Bornes C- | [1, 500] |
| Bornes γ | [10⁻⁴, 1] |

## Référence

Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002).
*A fast and elitist multiobjective genetic algorithm: NSGA-II.*
IEEE Transactions on Evolutionary Computation, 6(2), 182–197.
