import numpy        as np
import numpy.random as nr
from settings import Settings
import sys, csv, os

from optimization import *

def bitvec(rng, shape):
    return rng.integers(low=0, high=2, size=shape)

def mutate(rng, a, s):
    index = rng.integers(low=0, high=a.shape[0], size=1)[0]
    a[index] = rng.integers(low=0, high=s.base_max, size=1)

def biased_crossover(rng, a, b):
    return np.minimum(a + b, 1)

def random_crossover(rng, a, b):
    new  = np.zeros(a.shape, dtype=np.float32)
    mask = bitvec(rng, a.shape[0])
    new  += mask * a + (1 - mask) * b
    return new

def flip(rng, p):
    return rng.random() < p

def initialize(population, s):
    population[:, :s.size] = s.rng.integers(low=0, high=s.base_max,
            size=(s.pop_size, s.size))

def step(workspace, population, func, s):
    parents = s.rng.integers(low=0, high=s.pop_size, size=s.work_size)
    for i in range(s.pop_size):
        # Index into parents, selected uniformly
        ai = parents[2 * i]; bi = parents[2 * i + 1]
        a  = population[ai, :]; b = population[bi, :]
        age   = max(a[s.size + 1], b[s.size + 1]) # Inherit age
        child = random_crossover(s.rng, a[:s.size], b[:s.size])
        if flip(s.rng, s.mutation_probability): # Only mutate children
            mutate(s.rng, child, s)
        workspace[s.pop_size + i, :s.size] = child
        workspace[s.pop_size + i, s.size + 1] = age
    # Evaluate all fitnesses for population
    # Could be vectorized for certain fitness functions
    for i in range(s.work_size):
        workspace[i, s.size:-2] = fitness(workspace[i, :s.size], func, s)

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
    if s.age_fitness:
        workspace[:s.pop_size, s.size + s.objectives - 1] += 1
    # Otherwise, age is always zero and not considered

def fitness(v, func, s):
    xv, yv = np.split(v, np.array([s.member_size]))
    x = decode(xv, s); y = decode(yv, s)
    return func(x, y)

def decode(v, s):
    l = v.shape[0] - 1;
    return (-1. * max(v[0],1) +
            np.sum(v[1:] / (10.**np.arange(-s.pre_bits, s.post_bits))))

def age_fitness_pareto_optimization(func, settings):
    s = settings # Shorthand, conventional
    ''' Represent population as a large tensor for in-place ops
        Guaranteed to be slightly confusing, but fairly efficient.
        Workspace is a large matrix twice the population size, containing
        vectors for each member & all offspring, and then columns for
        fitness and age (or for the number of objectives) '''
    workspace  = np.zeros((s.work_size, s.entry), dtype=np.float32)
    population = workspace[:s.pop_size, :] # View, not copy
    initialize(population, settings)
    for g in range(settings.generations):
        # Crossover, mutation, aging, and fitness calculation:
        step(workspace, population, func, settings)
        # Age-fitness pareto selection process
        calculate_pareto_rank(workspace, settings)
        # Just sort by pareto rank (NSGA2)
        pareto_sort(workspace, settings)

def write_metrics():
    with open('metrics.csv', 'w', newline='') as metrics_file:
        metrics = csv.writer(metrics_file)
        metrics.writerow(['fitness'])
        metrics.writerow([np.mean(workspace[:s.pop_size, s.size])])

def main(args):
    # f_name = 'viennet'
    # f_name = 'sphere'
    # f_name = 'rosenbrock'
    f_name = 'rastrigrin'
    # f_name = 'ackley'
    # f_name = 'fonseca_fleming'
    func = all_functions[f_name]
    test = func(0, 0)
    try:
        objectives = len(test) + 1
    except TypeError:
        objectives = 2
    dimensions  = 2 # Higher dimensions not supported
    member_size = 8 + 1 # 4+8 bits for num, 1 bit for sign
    seed=2022
    settings = Settings(seed=seed, size=member_size*2, pop_size=100,
        member_size=member_size,
        pre_bits=0, post_bits=8,
        mutation_probability=0.1, objectives = objectives,
        base_max=10, # For base 10
        generations=100,
        age_fitness=True)
    settings.update(entry=settings.size + settings.objectives + 1,
        work_size = 2 * settings.pop_size,
        rng = nr.default_rng(settings.seed))
    age_fitness_pareto_optimization(func, settings)

if __name__ == '__main__':
    main(sys.argv[1:])
