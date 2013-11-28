import numpy as np

def competence_1(target, reached):
    return np.exp(min(-0.01, -np.linalg.norm(target-reached)))
