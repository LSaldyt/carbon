import sys, os

kidn = 'debug'
kind = 'release'
DIR = f'target/{kind}'
sys.path.insert(0, DIR)

import libcarbon as carbon

print(dir(carbon))
carbon.run_simple_ga(3)
