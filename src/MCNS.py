import math
import numpy as np
import random

#  Calculate the distance from one genome to another
def dist(g1 : np.ndarray, g2 : np.ndarray) -> float:
    return np.linalg.norm(g1-g2)

# Standard Uniform crossover
def uniform_crossover(g1 : np.ndarray, g2 : np.ndarray, random : random.Random = random) -> (np.ndarray, np.ndarray):
    indices = [random.randint(0,1) for _ in range(len(g1))]
    return np.array([g1[i] if x == 0 else g2[i] for i ,x in enumerate(indices)]), \
           np.array([g2[i] if x == 0 else g1[i] for i ,x in enumerate(indices)])

GENE_LIMIT = 8
def gene_limit(candidate : np.ndarray):
    return 1 - 0.25 * math.e ** ((candidate > 0).sum() - 5)

class NoveltySearch:

    def __init__(self, n : int, k : int, population_func, mutation_func, mc_criterion, novelty_threshold : float,
                 novelty_metric = dist, crossover_func = uniform_crossover, mu : float = 0.05, eval_refresh : int = 2500,
                 random : random.Random = random):
        # Standard EA Stuff
        self.n = n
        self.population_func = population_func
        self.crossover_func = crossover_func
        self.mutation_func = mutation_func
        self.mu = mu

        # Novelty Search Stuff
        self.k = k
        self.mc_criterion = mc_criterion
        self.pmin = novelty_threshold
        self.novelty_metric = novelty_metric

        self.eval_refresh = eval_refresh
        self.num_evals = 0
        self.num_added_to_archive = 0

        self.random = random

        self.archive = {}
        self.population = [self.population_func(random) for _ in range(self.n)]


    def run(self):
        # Random Selection
        self.random.shuffle(self.population)

        # Crossover
        offspring = []
        for i in range(0, self.n, 2):
            offspring.extend(self.crossover_func(self.population[i], self.population[2]))

        # Mutation
        for i, o in enumerate(offspring):
            if self.random.random() < self.mu:
                offspring[i] = self.mutation_func(o, self.random)

        # K Nearest Neighbours
        self.population.extend(offspring)

        eval_values = {}
        pop_plus_archive = self.population + [self.archive[g] for g in self.archive]
        for g1 in self.population:
            idx = frozenset(g1)
            # If genome meets minimum criterion
            if self.mc_criterion(g1):
                others = [g2 for g2 in pop_plus_archive if g1 is not g2]
                distances = {frozenset(g2) : self.novelty_metric(g1, g2) for g2 in others}
                def sort_key(o):
                    return distances[frozenset(o)]

                others.sort(key=sort_key)
                eval_values[idx] = sum([distances[frozenset(others[i])] for i in range(self.k)]) / self.k
            else:  # Else Novelty = 0.0
                eval_values[idx] = 0.0
            self.num_evals += 1

            # Modify repertoire
            if eval_values[idx] * gene_limit(g1) > self.pmin and idx not in self.archive:
                self.archive[idx] = g1
                self.num_added_to_archive += 1

            if self.num_evals % self.eval_refresh == 0:
                if self.num_added_to_archive > 4:
                    self.pmin += self.pmin * 0.2
                elif self.num_added_to_archive == 0:
                    self.pmin -= self.pmin * 0.05

                # Reset
                self.num_evals = 0
                self.num_added_to_archive = 0


        def sort_key(o):
            return eval_values[frozenset(o)]

        # Evaluation
        self.population.sort(key=sort_key, reverse=True)

        # Update new population
        self.population = self.population[:self.n]
