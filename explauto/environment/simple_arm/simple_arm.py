import numpy as np

# from pypot.kinematics import Link, Chain
# dim=2
# link1=Link(0, 0, 0.6, 0)
# link2=Link(0, 0, 0.4, 0)
# arm=Chain([link1, link2])

lengths = np.array([1])


def forward(angles, lengths=lengths):
    """ Link object as defined by the standard DH representation.
    :param list angles: angles of each joint
    """
    a = np.array(angles)
    a = np.cumsum(a)
    return sum(np.cos(a)*lengths), sum(np.sin(a)*lengths)

    # tr=arm.forward_kinematics(angles)
    # return tr[range(dim), -1]
