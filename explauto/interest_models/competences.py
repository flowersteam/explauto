import numpy as np

def competence_1(target, reached):
    return np.exp(min(-0.1, -np.linalg.norm(target-reached)))

def competence_bool(target, reached):
    return float((target == reached).all())
