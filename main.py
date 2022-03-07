from leaps.karel.generator import *
from leaps.karel.dsl import get_DSL
from age_pareto import main as afpo_main
import sys

def main(args):
    afpo_main(args)

if __name__ == '__main__':
    main(sys.argv[1:])
