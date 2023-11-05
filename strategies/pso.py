""" The strategy that uses particle swarm optimization"""
import sys
import random

import numpy as np


def tune(cost_func, num_particles=20, w=0.5, c1=2.0, c2=1.0,
         budget=-1, verbose=False):

    # init algorithm
    bounds = cost_func.bounds
    best_score_global = sys.float_info.max
    best_position_global = []
    budget = 50 * bounds.lb.shape[0]**2 if budget == -1 else budget

    # init particle swarm
    swarm = []
    for _ in range(0, num_particles):
        swarm.append(Particle(bounds))

    # start optimization
    while budget:
        if verbose:
            print("{} budgets remain, best score global: {}".format(
                budget, best_score_global))

        # evaluate particle positions
        for j in range(num_particles):
            swarm[j].evaluate(cost_func)

            # update global best if needed
            if swarm[j].score <= best_score_global:
                best_position_global = swarm[j].position
                best_score_global = swarm[j].score
            budget -= 1
            if budget == 0:
                break

        # update particle velocities and positions
        for j in range(0, num_particles):
            swarm[j].update_velocity(best_position_global, w, c1, c2)
            swarm[j].update_position(bounds)

    if verbose:
        print('Final result:')
        print(best_position_global)
        print(best_score_global)

    return cost_func.state


class Particle:
    def __init__(self, bounds):
        self.ndim = bounds.lb.shape[0]
        self.velocity = np.random.uniform(-1, 1, self.ndim)
        self.position = np.random.uniform(bounds.lb, bounds.ub)
        self.best_pos = self.position
        self.best_score = sys.float_info.max
        self.score = sys.float_info.max

    def evaluate(self, cost_func):
        self.score = cost_func(self.position)
        # update best_pos if needed
        if self.score < self.best_score:
            self.best_pos = self.position
            self.best_score = self.score

    def update_velocity(self, best_position_global, w, c1, c2):
        r1 = random.random()
        r2 = random.random()
        vc = c1 * r1 * (self.best_pos - self.position)
        vs = c2 * r2 * (best_position_global - self.position)
        self.velocity = w * self.velocity + vc + vs

    def update_position(self, bounds):
        self.position = self.position + self.velocity
        self.position = np.minimum(self.position, bounds.ub)
        self.position = np.maximum(self.position, bounds.lb)
