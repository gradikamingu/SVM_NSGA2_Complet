# Optimisation bi-critère SVM × NSGA-II
## AVC · Cancer du sein · Diabète

### Utilisation

**Placer le dataset AVC :**
```
data/healthcare-dataset-stroke-data.csv
```
Télécharger ici : https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
(Si absent, un dataset synthétique équivalent est utilisé automatiquement.)

**Lancer les expérimentations :**
```bash
pip install numpy pandas scikit-learn matplotlib
python run_experiments.py
```
Durée estimée : 8–15 minutes. Résultats dans `results_experiments/`.

### Structure
```
├── run_experiments.py          ← Point d'entrée principal
├── src/
│   ├── nsga2.py                ← NSGA-II from scratch
│   ├── svm_evaluator.py        ← Évaluation bi-critère (NFA, NFR)
│   └── visualisation.py        ← Figures dataset AVC seul
├── data/
│   ├── load_stroke.py          ← Chargeur dataset AVC réel + fallback
│   └── generate_dataset.py     ← Générateur synthétique (fallback)
└── experiments/
    ├── datasets.py             ← 3 datasets médicaux
    ├── competitors.py          ← 4 méthodes comparées
    ├── sensitivity_analysis.py ← Analyse N/M
    └── visualisation_experiments.py
```

### Datasets
| Dataset | Source | Lien |
|---|---|---|
| AVC | Kaggle (fedesoriano) | https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset |
| Cancer du sein | UCI / sklearn | https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic) |
| Diabète | UCI / Kaggle | https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database |
