import numpy        as np
import numpy.random as nr
from settings import Settings
from time import time
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


def random_pairs(s, high=None, size=None):
    size = size if size is not None else s.work_size
    high = high if high is not None else s.pop_size
    pairs = s.rng.integers(low=0, high=high, size=size)
    for i in range(size // 2):
        ai = pairs[2 * i]; bi = pairs[2 * i + 1]
        yield ai, bi

def step(workspace, population, func, s):
    for i, (ai, bi) in enumerate(random_pairs(s)):
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

def calculate_dominated(i, j, s, objectives):
    dominated = True
    for m in range(s.objectives):
        # Assume all objectives minimizing
        if objectives[i, m] < objectives[j, m]:
            dominated = False
    return dominated

def calculate_exact_pareto_rank(workspace, s):
    objectives = workspace[:, s.size:-1] # View
    dom_counts = workspace[:, -1]        # View
    # Brute force pareto front calculation :)
    for i in range(s.work_size):
        for j in range(s.work_size):
            if i == j:
                continue
            if calculate_dominated(i, j, s, objectives):
                dom_counts[i] += 1

def calculate_tournament_pareto_rank(workspace, s):
    objectives = workspace[:, s.size:-1] # View
    dom_counts = workspace[:, -1]        # View
    for i, j in random_pairs(s, size=s.work_size * s.estimate_factor, high=s.work_size):
        if i == j:
            continue
        if calculate_dominated(i, j, s, objectives):
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
    # Replace the worst n_new members with regenerated ones
    workspace[s.pop_size-s.n_new:s.pop_size, :s.size] = s.rng.integers(low=0, high=s.base_max,
            size=(s.n_new, s.size))

def fitness(v, func, s):
    xv, yv = np.split(v, np.array([s.member_size]))
    x, y = decode_section(v, s)
    return func(x, y)

def decode_section(v, s):
    xv, yv = np.split(v, np.array([s.member_size]))
    return decode(xv, s), decode(yv, s)

def decode(v, s):
    l = v.shape[0] - 1;
    return (-1. * max(v[0],1) +
            np.sum(v[1:] / (10.**np.arange(-s.pre_bits, s.post_bits))))

def age_fitness_pareto_optimization(func, metrics, long, settings):
    s = settings # Shorthand, conventional
    ''' Represent population as a large tensor for in-place ops
        Guaranteed to be slightly confusing, but fairly efficient.
        Workspace is a large matrix twice the population size, containing
        vectors for each member & all offspring, and then columns for
        fitness and age (or for the number of objectives) '''
    workspace  = np.zeros((s.work_size, s.entry), dtype=np.float32)
    population = workspace[:s.pop_size, :] # View, not copy
    initialize(population, settings)
    for g in range(s.generations):
        start = time()
        # Crossover, mutation, aging, and fitness calculation:
        step(workspace, population, func, settings)
        # Age-fitness pareto selection process
        if s.exact:
            # Just sort by pareto rank (NSGA2)
            calculate_exact_pareto_rank(workspace, settings)
        else:
            # Do tournament selection as in the original paper
            calculate_tournament_pareto_rank(workspace, settings)
        end = time()
        pareto_sort(workspace, settings)
        preamble = [g, s.seed, s.age_fitness, s.exact, end - start]
        metrics.writerow(preamble +
                [np.mean(workspace[:s.pop_size, s.size + o])
                 for o in range(s.objectives)])
        for m in range(s.pop_size):
            x, y = decode_section(population[m, :s.size], s)
            long.writerow(preamble + [x, y] +
                [population[m, s.size + o]
                 for o in range(s.objectives)])
    return metrics

def main(args):
    # f_name = 'viennet'
    # f_name = 'sphere'
    # f_name = 'rosenbrock'
    # f_name = 'rastrigrin'
    # f_name = 'ackley'
    # f_name = 'fonseca_fleming'
    # f_name = 'bihn_korn'
    # for f_name in all_functions:
    for f_name in ['rastrigrin']:
        print(f'Starting {f_name}')
        func = all_functions[f_name]
        test = func(0, 0)
        try:
            objectives = len(test) + 1
        except TypeError:
            objectives = 2
        dimensions  = 2 # Higher dimensions not supported
        member_size = 8 + 1 # 4+8 bits for num, 1 bit for sign
        seed=2022
        n_seeds = 32
        settings = Settings(seed=seed, size=member_size*2, pop_size=32,
            member_size=member_size,
            pre_bits=0, post_bits=8,
            mutation_probability=0.1, objectives = objectives,
            base_max=10, # For base 10
            generations=128,
            estimate_factor=16,
            n_new=4,
            exact=True,
            age_fitness=True)
        settings.update(entry=settings.size + settings.objectives + 1,
            work_size = 2 * settings.pop_size,
            rng = nr.default_rng(settings.seed))
        print(settings)
        pre = f'data/{f_name}'
        with open(f'{pre}_approx_metrics.csv', 'w', newline='') as metrics_file:
            metrics = csv.writer(metrics_file)
            preamble = ['generation', 'seed', 'age_enabled', 'exact', 'duration']
            fits = [f'f{i}' for i in range(objectives - 1)] + ['age']
            metrics.writerow(preamble + fits)
            with open(f'{pre}_approx_long.csv', 'w', newline='') as long_file:
                long = csv.writer(long_file)
                long.writerow(preamble + ['x', 'y'] + fits)
                for exact in (True,False):
                    for age_fitness in (True,False):
                        for seed in range(n_seeds):
                            print(f'{f_name}: exact={exact:5}, age={age_fitness:5}, seed={seed:3}', flush=True)
                            settings.update(seed=seed, exact=exact, age_fitness=age_fitness)
                            age_fitness_pareto_optimization(func, metrics, long, settings)

if __name__ == '__main__':
    main(sys.argv[1:])
