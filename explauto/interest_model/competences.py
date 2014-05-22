import numpy as np


def competence_dist(target, reached, dist_min=0.1):
    return min(- dist_min, - np.linalg.norm(target - reached))


def competence_exp(target, reached, dist_min=0.1):
    return np.exp(min(- dist_min, - np.linalg.norm(target - reached)))


def competence_bool(target, reached):
    return float((target == reached).all())
