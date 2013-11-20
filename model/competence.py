import numpy as np

def competence_1(target, reached):
    return min(-0.1, -np.linalg.norm(target-reached))
