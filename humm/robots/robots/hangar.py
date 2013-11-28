
    ### Function to create robots ###

def robotlist():
    return ('KinematicArm2D',
            'VowelModel',
            'Ergorobot',
            'Sinus1D',
            'RobotFunction'
           )


def robot(robotname, *args, **kwargs):
    """Return an instance of the robot specified by robotname"""
    return globals()[robotname](*args, **kwargs)


    ### List of available robots ###

"""
KinematicArm - Perfect kinematic arm controlled in position
    @param dimM            motor dimensionality : number of joints (default = 6)
    @param limits          angle limits for all joints in degrees (default : (-70, 70)
    @param joints_lenghts  a list of n_joints floats describing the rigid body lengths.
                           (default : 10.0 for each rigid body)
"""
from .multikin2d import KinematicArm2D


"""
VowelModel - Model of vowel generation and perception by de Boer:
B. de Boer, "The Origin of Vowel Systems", Oxford: Oxford University Press
"""
from .vocalizer import Vocalizer as VowelModel


"""
Sinus - A multidimensional function outputing in 1D, with configurable complexity
    @param dimM        the motor dimensionality (default: 1)
    @param complexity  the higher, the higher the learning complexity (i.e. samples to
                       construct good local models). (default: 1.0)
"""
from .fun import Sinus1D


"""
RobotFunction - A robot interface wrapping a user-supplied function.

    @param f       function taking at most one positional argument,
                   an array of float of dimension dimM, and returning
                   an array of float of dimension dimS
    @param bounds  describe the support of the function, as an hyperrectangle.
                   if not provided, each float will be constrained between
                   0 and 1.

"""
from .fun import RobotFunction


"""
Ergorobot - Forward model.
"""
from .ergorobot import Ergorobot

# A Dual wheel robot
from .dualwheel import DualWheel

from .prism import Prism, Prism1, Prism2, Prism3, Prism4
