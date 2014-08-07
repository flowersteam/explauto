from sympy import symbols
from sympy.physics.mechanics import *
from sympy import Dummy, lambdify
from numpy import array, hstack, zeros, linspace, cos, sin, arange
from numpy.linalg import solve
from scipy.integrate import odeint


def step(weights, duration):
    """ This function creates a sum of boxcar functions.

    :param numpy.array weight: height of each boxcar function.
    :param float duration: duration of the generated trajectory.
    """

    dt = duration / len(weights)

    def activate(t, dt):
        """This function returns 1 if t is in [0, dt[ and 0 otherwise.

        :param float t: current time
        :param float dt: time step
        """
        return 0 <= t < dt

    return (lambda t: sum([w * activate(t - i * dt, dt) for i, w in enumerate(weights)]))


def simulate(n, x0, dt, func):
    """ This function simulates the n-pendulum behavior.

    :param int n: number of particules suspended to the top one
    :param numpy.array x0: 2*(n+1)-length array, initial conditions (q and u) for each particle
    :param float dt: time step
    :param func func: input function that give forces along x_axis for each time t

    :rtype: array that contains all the coordinates and speeds of each point at time t.

    **Modifiable arguments**

    * arm_length (int): length of each segment
    * bob_mass (int): masse of each particle
    * t (array): time vector

    For more advanced setup, change the parameter_vals values. It contains the gravity, the mass and length of each particle.

    **Understand the code**

    The explanations below are for those who want to understand how this code works.
    For more detailed explanations please refer to http://www.moorepants.info/blog/npendulum.html

    **Variables meaning**

    - q : Generalized coordinates
    - u : Generalized speeds
    - f : Force applied to the cart
    - m : Mass of each bob
    - l : Length of each link
    - g : Gravity
    - t : Time
    - I : Inertial reference frame
    - O : Origin point. Its velocity is zero
    - P0 : Hinge point of top link
    - Pa0 : Particle at P0. Its position is q[0] and its velocity is q[0]
    - forces : List to hold the n + 1 applied forces, including the input force
    - kindiffs : List to hold kinematic ODE's

     In brief, we implement first the frames, points, particles and forces.
     Then we derive the equations of motion of the system thanks to KanesMethod.
     Next we define some constants that the user can change (mass, length, etc.), which allow us to set the differential equations, that are formatted by right_hand_side function.
     This function allows odeint to solve the differential equation.
     """

    def functor(f):
        """ This function allows us to give to :right_hand_side: function a specific input function.

        It is necessary because of the format of arguments of odeint.

        :param func f: input function

        :rtype: differential equation that will be solved.
        """
        def right_hand_side(x, t, args):
            arguments = hstack((x, f(t), args))
            dx = array(solve(M_func(*arguments), F_func(*arguments))).T[0]
            return dx
        return right_hand_side

    q = dynamicsymbols('q:' + str(n + 1))
    u = dynamicsymbols('u:' + str(n + 1))
    f = dynamicsymbols('f')
    m = symbols('m:' + str(n + 1))
    l = symbols('l:' + str(n))
    g, t = symbols('g t')
    I = ReferenceFrame('I')
    O = Point('O')
    O.set_vel(I, 0)
    P0 = Point('P0')
    P0.set_pos(O, q[0] * I.x)
    P0.set_vel(I, u[0] * I.x)
    Pa0 = Particle('Pa0', P0, m[0])
    frames = [I]
    points = [P0]
    particles = [Pa0]
    forces = [(P0, f * I.x - m[0] * g * I.y)]
    kindiffs = [q[0].diff(t) - u[0]]

    for i in range(n):
        Bi = I.orientnew('B' + str(i), 'Axis', [q[i + 1], I.z])
        Bi.set_ang_vel(I, u[i + 1] * I.z)
        frames.append(Bi)
        Pi = points[-1].locatenew('P' + str(i + 1), l[i] * Bi.x)
        Pi.v2pt_theory(points[-1], I, Bi)
        points.append(Pi)
        Pai = Particle('Pa' + str(i + 1), Pi, m[i + 1])
        particles.append(Pai)
        forces.append((Pi, -m[i + 1] * g * I.y))
        kindiffs.append(q[i + 1].diff(t) - u[i + 1])

    arm_length = 1. / n
    bob_mass = 0.01 / n
    parameters = [g, m[0]]
    parameter_vals = [9.81, 0.01 / n]
    for i in range(n):
        parameters += [l[i], m[i + 1]]
        parameter_vals += [arm_length, bob_mass]
    t = linspace(0, 1000 * dt, 1000)

    kane = KanesMethod(I, q_ind=q, u_ind=u, kd_eqs=kindiffs)
    fr, frstar = kane.kanes_equations(forces, particles)
    fr
    frstar

    dynamic = q + u
    dynamic.append(f)
    dummy_symbols = [Dummy() for i in dynamic]
    dummy_dict = dict(zip(dynamic, dummy_symbols))
    kindiff_dict = kane.kindiffdict()
    M = kane.mass_matrix_full.subs(kindiff_dict).subs(dummy_dict)
    F = kane.forcing_full.subs(kindiff_dict).subs(dummy_dict)
    M_func = lambdify(dummy_symbols + parameters, M)
    F_func = lambdify(dummy_symbols + parameters, F)

    f = functor(func)

    y = odeint(f, x0, t, args=(parameter_vals,))

    return y


def cartesian(n, states):
    """ This function computes cartesians coordinates from the states returned by the simulate function.

    :param int n: number of particules suspended to the top one
    :param numpy.array states: list of the positions and speeds at a certain time.
    """
    length = 1. / n  # arm_length
    pos_x = hstack((states[0], zeros(n)))
    pos_y = zeros(n + 1)
    for j in arange(1, n + 1):
        pos_x[j] = pos_x[j - 1] + length * cos(states[j])
        pos_y[j] = pos_y[j - 1] + length * sin(states[j])
    pos = hstack((pos_x, pos_y))
    return pos
