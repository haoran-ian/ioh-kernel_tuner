import random


def base_hillclimb(base_solution, cost_func, step_size,
                   budget, max_iter, greedy=True):
    best_solution = base_solution[:]
    best_score = cost_func(best_solution)
    budget -= 1

    for _ in range(max_iter):
        if budget == 0:
            return best_solution, best_score, budget
        candidate_solution = [
            x + random.uniform(-step_size, step_size) for x in best_solution]
        candidate_score = cost_func(candidate_solution)
        budget -= 1

        if candidate_score < best_score:
            best_solution, best_score = candidate_solution, candidate_score
            if greedy:
                return best_solution, best_score, budget

    return best_solution, best_score, budget
