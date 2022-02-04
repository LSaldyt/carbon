import sys, os

DIR = 'target/release'
sys.path.insert(0, DIR)

import libmeta_rust as metarust

print(metarust.sum_as_string(1, 2))
