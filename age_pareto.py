import numpy        as np
import numpy.random as nr
from settings import Settings
import sys, os

def bitvec(rng, shape):
    return rng.integers(low=0, high=2, size=shape)

def mutate(rng, a):
    index = rng.integers(low=0, high=a.shape[0], size=1)[0]
    a[index] = bitvec(rng, 1)

def biased_crossover(rng, a, b):
    return np.minimum(a + b, 1)

def random_crossover(rng, a, b):
    new  = np.zeros(a.shape, dtype=np.int32)
    mask = bitvec(rng, a.shape[0])
    new  += mask * a + (1 - mask) * b
    return new

def relative_fitness(member, target):
    return np.sum(np.abs(target - member)) # L1

def flip(rng, p):
    return rng.random() < p

def initialize(population, s):
    population[:, :s.size] = bitvec(s.rng, (s.pop_size, s.size))

def step(workspace, population, target, s):
    parents = s.rng.integers(low=0, high=s.pop_size, size=s.work_size)
    for i in range(s.pop_size):
        # Index into parents, selected uniformly
        ai = parents[2 * i]; bi = parents[2 * i + 1]
        a  = population[ai, :]; b = population[bi, :]
        age   = max(a[s.size + 1], b[s.size + 1]) # Inherit age
        child = random_crossover(s.rng, a[:s.size], b[:s.size])
        if flip(s.rng, s.mutation_probability): # Only mutate children
            mutate(s.rng, child)
        workspace[s.pop_size + i, :s.size] = child
        workspace[s.pop_size + i, s.size + 1] = age
    # Evaluate all fitnesses for population
    # Could be vectorized for certain fitness functions
    for i in range(s.work_size):
        workspace[i, s.size] = relative_fitness(workspace[i, :s.size],
                                                target)

def calculate_pareto_rank(workspace, s):
    objectives = workspace[:, s.size:-1] # View
    dom_counts = workspace[:, -1]        # View
    # Brute force pareto front calculation :)
    for i in range(s.work_size):
        for j in range(s.work_size):
            if i == j:
                continue
            dominated = True
            for m in range(s.objectives):
                # Assume all objectives minimizing
                if objectives[i, m] < objectives[j, m]:
                    dominated = False
            if dominated:
                dom_counts[i] += 1

def pareto_sort(workspace, s):
    dom_counts = workspace[:, -1] # View
    rank_index = dom_counts.argsort()
    workspace[:] = workspace[rank_index]
    workspace[:, -1] = 0 # Reset dom counts
    # Increment age for the selected members
    workspace[:s.pop_size, s.size + s.objectives - 1] += 1

def age_fitness_pareto_optimization(settings):
    s = settings # Shorthand, conventional
    ''' Represent population as a large tensor for in-place ops
        Guaranteed to be slightly confusing, but fairly efficient.
        Workspace is a large matrix twice the population size, containing
        bitvectors for each member & all offspring, and then columns for
        fitness and age (or for the number of objectives) '''
    workspace  = np.zeros((s.work_size, s.entry), dtype=np.int32)
    population = workspace[:s.pop_size, :] # View, not copy
    # print(workspace)
    initialize(population, settings)
    # print(workspace)
    # Start by evolving bit vectors which match a target
    target = bitvec(s.rng, s.size)
    for g in range(settings.generations):
        # Crossover, mutation, aging, and fitness calculation:
        step(workspace, population, target, settings)
        # print(workspace)
        # Age-fitness pareto selection process:
        calculate_pareto_rank(workspace, settings)
        # Just sort by pareto rank (NSGA2)
        pareto_sort(workspace, settings)
        print(np.mean(workspace[:, -3]))

def main(args):
    settings = Settings(seed = 2022, size = 100, pop_size = 10,
        mutation_probability = 0.1, objectives = 2,
        generations=100)
    settings.update(entry=settings.size + settings.objectives + 1,
        work_size = 2 * settings.pop_size,
        rng = nr.default_rng(settings.seed))
    age_fitness_pareto_optimization(settings)

if __name__ == '__main__':
    main(sys.argv[1:])
