"""The strategy that uses multi-start local search."""
from strategies.greedy_mls import tune as mls_tune


def tune(cost_func, budget=-1, max_iter=100, greedy=False, verbose=False):

    # Default MLS uses 'best improvement' hillclimbing, so greedy hillclimbing
    # is disabled with greedy defaulting to False.
    return mls_tune(cost_func, budget, max_iter, greedy, verbose)
