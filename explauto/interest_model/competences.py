import numpy as np


def competence_dist(target, reached, dist_min=0., dist_max=1.):
    return max(- dist_max, min(- dist_min, - np.linalg.norm(target - reached)))


def competence_exp(target, reached, dist_min=0., dist_max=1., power=1.):
    return np.exp(power * competence_dist(target, reached, dist_min, dist_max))


def competence_bool(target, reached):
    return float((target == reached).all())
