"""
Implémentation de l'algorithme NSGA-II (Non-dominated Sorting Genetic Algorithm II)
pour l'optimisation bi-critère des hyper-paramètres SVM.

Référence : Deb et al., "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II",
            IEEE Transactions on Evolutionary Computation, 2002.

Structure d'un individu :
    genes    : vecteur réel [C+, C-, gamma]
    objectives : [NFA, NFR] mesurés sur le jeu de validation
    rank     : rang de dominance (1 = front de Pareto)
    distance : crowding distance
"""

import numpy as np


# ─────────────────────────────────────────────
#  Dominance et tri de non-dominance
# ─────────────────────────────────────────────

def dominates(obj_a, obj_b):
    """
    Retourne True si la solution A domine la solution B.
    A domine B si A est au moins aussi bonne sur tous les objectifs
    et strictement meilleure sur au moins un.
    (ici on minimise NFA et NFR)
    """
    return (all(a <= b for a, b in zip(obj_a, obj_b)) and
            any(a < b for a, b in zip(obj_a, obj_b)))


def fast_non_dominated_sort(population):
    """
    Tri de non-dominance rapide (O(MN²)).
    Retourne une liste de fronts [F1, F2, ...], chaque front
    étant une liste d'indices dans la population.
    """
    n = len(population)
    domination_count = np.zeros(n, dtype=int)   # combien de solutions dominent i
    dominated_by = [[] for _ in range(n)]        # solutions que i domine
    fronts = [[]]

    for i in range(n):
        for j in range(i + 1, n):
            obj_i = population[i]['objectives']
            obj_j = population[j]['objectives']
            if dominates(obj_i, obj_j):
                dominated_by[i].append(j)
                domination_count[j] += 1
            elif dominates(obj_j, obj_i):
                dominated_by[j].append(i)
                domination_count[i] += 1

    for i in range(n):
        if domination_count[i] == 0:
            population[i]['rank'] = 1
            fronts[0].append(i)

    current_front = 0
    while fronts[current_front]:
        next_front = []
        for i in fronts[current_front]:
            for j in dominated_by[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    population[j]['rank'] = current_front + 2
                    next_front.append(j)
        current_front += 1
        fronts.append(next_front)

    return [f for f in fronts if f]  # supprimer le dernier front vide


def crowding_distance_assignment(population, front_indices):
    """
    Calcule la crowding distance pour les individus d'un front.
    Les individus aux extrémités reçoivent une distance infinie.
    """
    n = len(front_indices)
    if n == 0:
        return
    for i in front_indices:
        population[i]['distance'] = 0.0

    n_obj = len(population[front_indices[0]]['objectives'])
    for m in range(n_obj):
        # Trier par objectif m
        sorted_idx = sorted(front_indices,
                            key=lambda i: population[i]['objectives'][m])
        obj_min = population[sorted_idx[0]]['objectives'][m]
        obj_max = population[sorted_idx[-1]]['objectives'][m]

        # Les bornes ont une distance infinie
        population[sorted_idx[0]]['distance'] = float('inf')
        population[sorted_idx[-1]]['distance'] = float('inf')

        if obj_max == obj_min:
            continue
        for k in range(1, n - 1):
            population[sorted_idx[k]]['distance'] += (
                (population[sorted_idx[k + 1]]['objectives'][m] -
                 population[sorted_idx[k - 1]]['objectives'][m])
                / (obj_max - obj_min)
            )


def crowded_comparison(ind_a, ind_b):
    """
    Relation d'ordre partiel <n basée sur le rang et la crowding distance.
    Retourne True si ind_a est "meilleur" que ind_b.
    """
    return (ind_a['rank'] < ind_b['rank']) or \
           (ind_a['rank'] == ind_b['rank'] and
            ind_a['distance'] > ind_b['distance'])


# ─────────────────────────────────────────────
#  Opérateurs génétiques
# ─────────────────────────────────────────────

def tournament_selection(population, rng):
    """Sélection par tournoi binaire."""
    a, b = rng.choice(len(population), size=2, replace=False)
    winner = a if crowded_comparison(population[a], population[b]) else b
    return population[winner]['genes'].copy()


def sbx_crossover(parent1, parent2, bounds, rng, eta_c=20):
    """
    Simulated Binary Crossover (SBX).
    eta_c : paramètre de distribution (plus grand = enfants proches des parents).
    """
    child1 = parent1.copy()
    child2 = parent2.copy()

    for i in range(len(parent1)):
        if rng.random() > 0.5:
            continue
        if abs(parent1[i] - parent2[i]) < 1e-10:
            continue

        x1 = min(parent1[i], parent2[i])
        x2 = max(parent1[i], parent2[i])
        lo, hi = bounds[i]

        beta = 1.0 + 2.0 * min(x1 - lo, hi - x2) / (x2 - x1)
        alpha = 2.0 - beta ** (-(eta_c + 1))
        u = rng.random()

        if u <= 1.0 / alpha:
            beta_q = (u * alpha) ** (1.0 / (eta_c + 1))
        else:
            beta_q = (1.0 / (2.0 - u * alpha)) ** (1.0 / (eta_c + 1))

        child1[i] = 0.5 * ((x1 + x2) - beta_q * (x2 - x1))
        child2[i] = 0.5 * ((x1 + x2) + beta_q * (x2 - x1))

        child1[i] = np.clip(child1[i], lo, hi)
        child2[i] = np.clip(child2[i], lo, hi)

    return child1, child2


def polynomial_mutation(genes, bounds, rng, eta_m=20, mutation_prob=None):
    """
    Mutation polynomiale.
    mutation_prob : probabilité par gène (défaut 1/n_genes).
    """
    n = len(genes)
    if mutation_prob is None:
        mutation_prob = 1.0 / n
    mutated = genes.copy()

    for i in range(n):
        if rng.random() > mutation_prob:
            continue
        lo, hi = bounds[i]
        delta = hi - lo
        if delta < 1e-10:
            continue
        u = rng.random()
        if u < 0.5:
            delta_q = (2 * u) ** (1.0 / (eta_m + 1)) - 1.0
        else:
            delta_q = 1.0 - (2 * (1 - u)) ** (1.0 / (eta_m + 1))
        mutated[i] = np.clip(mutated[i] + delta_q * delta, lo, hi)

    return mutated


def create_offspring(population, bounds, rng, pop_size):
    """Crée une population fille Qt de taille pop_size."""
    offspring = []
    while len(offspring) < pop_size:
        p1 = tournament_selection(population, rng)
        p2 = tournament_selection(population, rng)
        c1, c2 = sbx_crossover(p1, p2, bounds, rng)
        c1 = polynomial_mutation(c1, bounds, rng)
        c2 = polynomial_mutation(c2, bounds, rng)
        offspring.append({'genes': c1, 'objectives': None,
                          'rank': None, 'distance': 0.0})
        if len(offspring) < pop_size:
            offspring.append({'genes': c2, 'objectives': None,
                              'rank': None, 'distance': 0.0})
    return offspring[:pop_size]


# ─────────────────────────────────────────────
#  Boucle principale NSGA-II
# ─────────────────────────────────────────────

def nsga2(evaluate_fn, bounds, pop_size=40, n_generations=50,
          random_state=42, verbose=True):
    """
    Algorithme NSGA-II.

    Paramètres
    ----------
    evaluate_fn : callable
        Prend un vecteur de gènes (C+, C-, gamma) et retourne [NFA, NFR].
    bounds : list of (lo, hi)
        Bornes pour chaque gène : [(C+_min, C+_max), (C-_min, C-_max), (gamma_min, gamma_max)].
    pop_size : int
        Taille de la population N.
    n_generations : int
        Nombre de générations M.
    random_state : int
        Graine aléatoire.
    verbose : bool
        Afficher la progression.

    Retourne
    --------
    pareto_front : list of dict
        Solutions non dominées de la population finale.
    history : list
        Historique du front de Pareto à chaque génération.
    """
    rng = np.random.RandomState(random_state)

    # ── Initialisation P0 ──
    if verbose:
        print("=" * 60)
        print("  NSGA-II : Optimisation bi-critère des hyper-paramètres SVM")
        print("=" * 60)
        print(f"  Population : {pop_size}  |  Générations : {n_generations}")
        print(f"  Bornes : C+ {bounds[0]}, C- {bounds[1]}, γ {bounds[2]}")
        print("-" * 60)

    def random_individual():
        genes = np.array([rng.uniform(lo, hi) for lo, hi in bounds])
        return {'genes': genes, 'objectives': None, 'rank': None, 'distance': 0.0}

    population = [random_individual() for _ in range(pop_size)]

    # Évaluation initiale
    for ind in population:
        ind['objectives'] = evaluate_fn(ind['genes'])

    # Tri initial
    fronts = fast_non_dominated_sort(population)
    for front in fronts:
        crowding_distance_assignment(population, front)

    history = []

    # ── Boucle principale ──
    for gen in range(n_generations):
        # Créer Qt (population fille)
        offspring = create_offspring(population, bounds, rng, pop_size)
        for ind in offspring:
            ind['objectives'] = evaluate_fn(ind['genes'])

        # Rt = Pt ∪ Qt
        combined = population + offspring

        # Tri de non-dominance sur Rt
        fronts = fast_non_dominated_sort(combined)
        for front in fronts:
            crowding_distance_assignment(combined, front)

        # Construire Pt+1
        new_population = []
        for front in fronts:
            if len(new_population) + len(front) <= pop_size:
                new_population.extend([combined[i] for i in front])
            else:
                # Remplir avec les meilleurs selon crowding distance
                remaining = pop_size - len(new_population)
                sorted_front = sorted(front,
                    key=lambda i: combined[i]['distance'], reverse=True)
                new_population.extend([combined[i] for i in sorted_front[:remaining]])
                break

        population = new_population

        # Enregistrer le front de Pareto courant
        pareto = [ind for ind in population if ind['rank'] == 1]
        nfa_vals = [ind['objectives'][0] for ind in pareto]
        nfr_vals = [ind['objectives'][1] for ind in pareto]

        history.append({
            'generation': gen + 1,
            'pareto_size': len(pareto),
            'nfa_min': min(nfa_vals) if nfa_vals else None,
            'nfr_min': min(nfr_vals) if nfr_vals else None,
        })

        if verbose and (gen + 1) % 10 == 0:
            print(f"  Génération {gen + 1:3d}/{n_generations} | "
                  f"Front de Pareto : {len(pareto)} solutions | "
                  f"NFA_min={min(nfa_vals):5.1f} | NFR_min={min(nfr_vals):5.1f}")

    # Front de Pareto final
    pareto_front = [ind for ind in population if ind['rank'] == 1]
    pareto_front.sort(key=lambda ind: ind['objectives'][0])  # trier par NFA

    if verbose:
        print("-" * 60)
        print(f"  Terminé. Front de Pareto final : {len(pareto_front)} solutions.")
        print("=" * 60)

    return pareto_front, history
