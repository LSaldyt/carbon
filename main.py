import sys, os

kind = 'debug'
kind = 'release'
DIR = f'target/{kind}'
sys.path.insert(0, DIR)

import libcarbon as carbon
from time import time

print(dir(carbon))
start = time()
carbon.run_generic_ga(iterations=10000, k=30, length=5,
                      mut_rate=1.0, cross_rate=1.0, elitism=1,
                      minimizing=False, init_rand=False,
                      pop_size=100, metrics_filename='metrics.csv')
end   = time()
print(f'Duration: {(end - start):0.6f}')
