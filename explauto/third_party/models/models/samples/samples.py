# Generate sample dataset to test regression algorithms.

import math
from math import cos, sin
import random

from .. import dataset

# Constant values
a = 35
b = 82
c = 80
d = 65
e = 67
f = 63

def forwardkin_6dof3d(t):
   t0, t1, t2, t3, t4, t5 = t
   x =(0
       - b*sin(t1)*cos(t0)
       + 0
       + -c*sin(t1 + t2)*cos(t0) - d*sin(t0)*sin(t3) + d*cos(t0)*cos(t3)*cos(t1 + t2)
       + e*(-sin(t0)*sin(t3)*cos(t4) - sin(t4)*sin(t1 + t2)*cos(t0) + cos(t0)*cos(t3)*cos(t4)*cos(t1 + t2))
       + f*(-sin(t0)*sin(t3)*cos(t4 + t5) - sin(t1 + t2)*sin(t4 + t5)*cos(t0) + cos(t0)*cos(t3)*cos(t1 + t2)*cos(t4 + t5))
       )
    
   y =(0
       - b*sin(t0)*sin(t1)
       + 0
       + -c*sin(t0)*sin(t1 + t2) + d*sin(t0)*cos(t3)*cos(t1 + t2) + d*sin(t3)*cos(t0)
       + e*(-sin(t0)*sin(t4)*sin(t1 + t2) + sin(t0)*cos(t3)*cos(t4)*cos(t1 + t2) + sin(t3)*cos(t0)*cos(t4))
       + f*(-sin(t0)*sin(t1 + t2)*sin(t4 + t5) + sin(t0)*cos(t3)*cos(t1 + t2)*cos(t4 + t5) + sin(t3)*cos(t0)*cos(t4 + t5))
       )
   
   z =(a
       + b*cos(t1)
       + 0
       + c*cos(t1 + t2) + d*sin(t1 + t2)*cos(t3)
       + e*(sin(t4)*cos(t1 + t2) + sin(t1 + t2)*cos(t3)*cos(t4))
       + f*(sin(t1 + t2)*cos(t3)*cos(t4 + t5) + sin(t4 + t5)*cos(t1 + t2))
       )
   
   return (x, y, z)




def sample_6dof3d(n):
    """Generate a sample dataset for a 6 DOFs arm kinematic in 3D space.
    The 6 DOFs arm is always the same, but the set change randomly.
    
    @param n  the number of samples
    @return   dataset, generative function.
    """
    dset = dataset.Dataset(6, 3)
    for i in range(n):
        x = tuple(random.uniform(-math.pi, math.pi) for _ in range(6))
        dset.add_xy(x, forwardkin_6dof3d(x))
    return dset, forwardkin_6dof3d, 6*((-math.pi, math.pi),)
