"""A greedy multi-start local search algorithm for parameter search."""
import sys
import numpy as np
from strategies.hillclimbers import base_hillclimb


def tune(cost_func, budget=-1, max_iter=100, greedy=True, verbose=False):

    # init algorithm
    bounds = cost_func.bounds
    best_score_global = sys.float_info.max
    best_position_global = np.random.uniform(bounds.lb, bounds.ub)
    budget = 50 * bounds.lb.shape[0]**2 if budget == -1 else budget

    # while searching
    while budget:
        if verbose:
            print("{} budgets remain, best score global: {}".format(
                budget, best_score_global))

        best_position_global, best_score_global, budget = base_hillclimb(
            best_position_global, cost_func, 0.1, budget, max_iter, greedy)

    if verbose:
        print('Final result:')
        print(best_position_global)
        print(best_score_global)

    return cost_func.state
