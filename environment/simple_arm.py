import configure
import kinematics as kin

dim=2

link1=kin.Link(0, 0, 0.6, 0)

link2=kin.Link(0, 0, 0.4, 0)

arm=kin.Chain([link1, link2])

def forward(angles):
    """ Link object as defined by the standard DH representation.
    :param list angles: angles of each joint
    """
    tr=arm.forward_kinematics(angles)
    return tr[range(dim), -1]
