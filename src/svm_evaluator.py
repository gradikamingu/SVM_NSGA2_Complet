"""
Évaluation bi-critère du modèle SVM.
NFA = Faux Positifs (sains classés AVC)
NFR = Faux Négatifs (AVC classés sains)
"""

import warnings
import numpy as np

# Suppression globale de TOUS les avertissements sklearn
warnings.filterwarnings("ignore")

from sklearn.svm import SVC


class SVMEvaluator:

    def __init__(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val   = X_val
        self.y_val   = y_val
        self.n_pos   = np.sum(y_val == 1)
        self.n_neg   = np.sum(y_val == -1)
        self._cache  = {}

    def _train_svc(self, C_pos, C_neg, gamma):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = SVC(
                C=1.0,
                kernel='rbf',
                gamma=gamma,
                class_weight={1: C_pos, -1: C_neg},
                random_state=0,
                max_iter=5000
            )
            clf.fit(self.X_train, self.y_train)
        return clf

    def evaluate(self, genes):
        C_pos, C_neg, gamma = genes
        key = (round(C_pos, 4), round(C_neg, 4), round(gamma, 6))
        if key in self._cache:
            return self._cache[key]
        try:
            clf    = self._train_svc(C_pos, C_neg, gamma)
            y_pred = clf.predict(self.X_val)
        except Exception:
            result = [float(self.n_neg), float(self.n_pos)]
            self._cache[key] = result
            return result

        NFA = float(np.sum((self.y_val == -1) & (y_pred == 1)))
        NFR = float(np.sum((self.y_val == 1)  & (y_pred == -1)))
        result = [NFA, NFR]
        self._cache[key] = result
        return result

    def full_metrics(self, genes):
        C_pos, C_neg, gamma = genes
        clf    = self._train_svc(C_pos, C_neg, gamma)
        y_pred = clf.predict(self.X_val)

        TP = np.sum((self.y_val == 1)  & (y_pred == 1))
        FP = np.sum((self.y_val == -1) & (y_pred == 1))
        TN = np.sum((self.y_val == -1) & (y_pred == -1))
        FN = np.sum((self.y_val == 1)  & (y_pred == -1))

        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        precision   = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        accuracy    = (TP + TN) / len(self.y_val)
        f1          = (2 * precision * sensitivity / (precision + sensitivity)
                       if (precision + sensitivity) > 0 else 0.0)

        return {
            'C+': C_pos, 'C-': C_neg, 'gamma': gamma,
            'TP': int(TP), 'FP': int(FP), 'TN': int(TN), 'FN': int(FN),
            'NFA': int(FP), 'NFR': int(FN),
            'Sensibilité (Rappel)': round(sensitivity, 4),
            'Spécificité':         round(specificity, 4),
            'Précision':           round(precision, 4),
            'Exactitude':          round(accuracy, 4),
            'Score F1':            round(f1, 4),
        }
