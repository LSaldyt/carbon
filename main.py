import sys, os

kind = 'debug'
kind = 'release'
DIR = f'target/{kind}'
sys.path.insert(0, DIR)

import libcarbon as carbon
from time import time

print(dir(carbon))
start = time()
carbon.run_simple_ga(1000, 'metrics.csv')
end   = time()
print(f'Duration: {(end - start):0.6f}')
