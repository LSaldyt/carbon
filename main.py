from leaps.karel.generator import *
from toy_afpo import main as afpo_main
import sys

def main(args):
    afpo_main(args)

if __name__ == '__main__':
    main(sys.argv[1:])
