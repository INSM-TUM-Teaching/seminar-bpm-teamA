# candidate.py
import random
import numpy as np

# Global definitions
POPULATION_SIZE = 50
N_GENERATIONS = 50
GENE_DIM = 12

# Define bounds for each gene
LOWER_BOUNDS = [24, 24] + [0] * 10
UPPER_BOUNDS = [168, 168] + [1] * 10

class Candidate:
    def __init__(self, gene=None):
        if gene is None:
            self.gene = [0] * GENE_DIM
        else:
            self.gene = gene
        self.fitness = None  # Tuple: (WTA, WTH, NERV, COST)
        self.rank = None
        self.crowding_distance = 0

def repair_solution(gene):
    # Ensure planning offsets (genes 0 and 1) are within bounds and p2 <= p1.
    p1, p2 = gene[0], gene[1]
    p1 = max(min(p1, UPPER_BOUNDS[0]), LOWER_BOUNDS[0])
    p2 = max(min(p2, UPPER_BOUNDS[1]), LOWER_BOUNDS[1])
    if p2 > p1:
        p2 = p1
    gene[0], gene[1] = p1, p2
    # Ensure scheduling fractions (genes 2 to 11) are in [0, 1]
    for i in range(2, GENE_DIM):
        gene[i] = min(max(gene[i], 0), 1)
    return gene

def random_candidate():
    gene = []
    for i in range(GENE_DIM):
        gene.append(random.uniform(LOWER_BOUNDS[i], UPPER_BOUNDS[i]))
    candidate = Candidate(gene)
    candidate.gene = repair_solution(candidate.gene)
    return candidate

def dominates(fitness1, fitness2):
    """Returns True if fitness1 dominates fitness2 (minimization objectives)."""
    less_or_equal = all(f1 <= f2 for f1, f2 in zip(fitness1, fitness2))
    strictly_less = any(f1 < f2 for f1, f2 in zip(fitness1, fitness2))
    return less_or_equal and strictly_less

def non_dominated_sort(population):
    """Performs non-dominated sorting on the population and returns a list of fronts."""
    fronts = []
    S = {}
    n = {}
    for p in population:
        S[p] = []
        n[p] = 0
        for q in population:
            if p == q:
                continue
            if dominates(p.fitness, q.fitness):
                S[p].append(q)
            elif dominates(q.fitness, p.fitness):
                n[p] += 1
        if n[p] == 0:
            p.rank = 0
    first_front = [p for p in population if n[p] == 0]
    fronts.append(first_front)
    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    q.rank = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    fronts.pop()  # Remove last empty list.
    return fronts

def calculate_crowding_distance(front):
    l = len(front)
    if l == 0:
        return
    m = len(front[0].fitness)
    for p in front:
        p.crowding_distance = 0
    for i in range(m):
        front.sort(key=lambda x: x.fitness[i])
        front[0].crowding_distance = float('inf')
        front[-1].crowding_distance = float('inf')
        f_min = front[0].fitness[i]
        f_max = front[-1].fitness[i]
        if f_max == f_min:
            continue
        for j in range(1, l - 1):
            front[j].crowding_distance += (front[j + 1].fitness[i] - front[j - 1].fitness[i]) / (f_max - f_min)

def tournament_selection(population):
    selected = []
    for _ in range(len(population)):
        a = random.choice(population)
        b = random.choice(population)
        if a.rank < b.rank:
            selected.append(a)
        elif a.rank > b.rank:
            selected.append(b)
        else:
            if a.crowding_distance > b.crowding_distance:
                selected.append(a)
            else:
                selected.append(b)
    return selected

def sbx_crossover(parent1, parent2, eta_c=15, crossover_prob=0.9):
    n_dim = len(parent1.gene)
    offspring1_gene = parent1.gene.copy()
    offspring2_gene = parent2.gene.copy()
    if random.random() <= crossover_prob:
        for i in range(n_dim):
            u = random.random()
            if u <= 0.5:
                beta = (2 * u) ** (1 / (eta_c + 1))
            else:
                beta = (1 / (2 * (1 - u))) ** (1 / (eta_c + 1))
            offspring1_gene[i] = 0.5 * ((1 + beta) * parent1.gene[i] + (1 - beta) * parent2.gene[i])
            offspring2_gene[i] = 0.5 * ((1 - beta) * parent1.gene[i] + (1 + beta) * parent2.gene[i])
    offspring1_gene = repair_solution(offspring1_gene)
    offspring2_gene = repair_solution(offspring2_gene)
    return Candidate(offspring1_gene), Candidate(offspring2_gene)

def polynomial_mutation(candidate, eta_m=20, mutation_prob=None):
    n_dim = len(candidate.gene)
    if mutation_prob is None:
        mutation_prob = 1.0 / n_dim
    mutated_gene = candidate.gene.copy()
    for i in range(n_dim):
        if random.random() <= mutation_prob:
            u = random.random()
            # Calculate delta based on u and eta_m
            if u < 0.5:
                delta = (2 * u) ** (1 / (eta_m + 1)) - 1
            else:
                delta = 1 - (2 * (1 - u)) ** (1 / (eta_m + 1))
            mutated_gene[i] += delta * (UPPER_BOUNDS[i] - LOWER_BOUNDS[i])
            mutated_gene[i] = min(max(mutated_gene[i], LOWER_BOUNDS[i]), UPPER_BOUNDS[i])
    mutated_gene = repair_solution(mutated_gene)
    return Candidate(mutated_gene)
