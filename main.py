from leaps.karel.generator import *
from leaps.karel.dsl import get_DSL
from age_pareto import main as afpo_main
import sys

def main(args):
    dsl = get_DSL(dsl_type='prob', seed=32, environment='karel')
    print(dir(dsl))
    tokens = dsl.random_tokens(start_token='prog',
                               max_depth=3,
                               max_nesting_depth=3)
    print(tokens)
    mutate = dsl.random_tokens(start_token='stmt',
                               max_depth=3,
                               max_nesting_depth=3)
    print(mutate)
    1/0
    random_code = dsl.random_code(max_depth=3,
                                  max_nesting_depth=3)
    print(random_code)
    print(dir(random_code))
    # afpo_main(args)

if __name__ == '__main__':
    main(sys.argv[1:])
