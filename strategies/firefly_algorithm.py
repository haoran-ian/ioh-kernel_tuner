"""The strategy that uses the firefly algorithm for optimization."""
import sys
import numpy as np
from strategies.pso import Particle


def tune(cost_func, num_particles=20, B0=1.0, gamma=1.0, alpha=0.2,
         budget=-1, verbose=False):

    # using this instead of get_bounds because scaling is used
    bounds = cost_func.bounds
    best_score_global = sys.float_info.max
    best_position_global = []
    budget = 50 * bounds.lb.shape[0]**2 if budget == -1 else budget

    # init particle swarm
    swarm = []
    for i in range(0, num_particles):
        swarm.append(Firefly(bounds))

    # compute initial intensities
    for j in range(num_particles):
        swarm[j].compute_intensity(cost_func)
        if swarm[j].score <= best_score_global:
            best_position_global = swarm[j].position
            best_score_global = swarm[j].score
        budget -= 1
        if budget == 0:
            break

    while budget:
        if verbose:
            print("{} budgets remain, best score global: {}".format(
                budget, best_score_global))

        # compare all to all and compute attractiveness
        for i in range(num_particles):
            if budget == 0:
                break
            for j in range(num_particles):
                if swarm[i].intensity < swarm[j].intensity:
                    dist = swarm[i].distance_to(swarm[j])
                    beta = B0 * np.exp(-gamma * dist * dist)

                    swarm[i].move_towards(swarm[j], beta, alpha)
                    swarm[i].compute_intensity(cost_func)

                    # update global best if needed, actually only used for printing
                    if swarm[i].score <= best_score_global:
                        best_position_global = swarm[i].position
                        best_score_global = swarm[i].score
                    budget -= 1
                    if budget == 0:
                        break

        swarm.sort(key=lambda x: x.score)

    if verbose:
        print('Final result:')
        print(best_position_global)
        print(best_score_global)

    return cost_func.state


class Firefly(Particle):
    """Firefly object for use in the Firefly Algorithm."""

    def __init__(self, bounds):
        """Create Firefly at random position within bounds."""
        super().__init__(bounds)
        self.bounds = bounds
        self.intensity = 1 / self.score

    def distance_to(self, other):
        """Return Euclidian distance between self and other Firefly."""
        return np.linalg.norm(self.position-other.position)

    def compute_intensity(self, fun):
        """Evaluate cost function and compute intensity at this position."""
        self.evaluate(fun)
        if self.score == sys.float_info.max:
            self.intensity = -sys.float_info.max
        else:
            self.intensity = 1 / self.score

    def move_towards(self, other, beta, alpha):
        """Move firefly towards another given beta and alpha values."""
        self.position += beta * (other.position - self.position)
        self.position += alpha * \
            (np.random.uniform(-0.5, 0.5, len(self.position)))
        self.position = np.minimum(self.position, self.bounds.ub)
        self.position = np.maximum(self.position, self.bounds.lb)
