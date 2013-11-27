# -*- coding: utf-8 -*-
__author__ = "Fabien Benureau"
__date__   = "06/2012"

"""Simple kinematic model for a two wheeled robot (wheelchair-style)
   with a sensor/motor interface.
    A dual accelerometer robot is provided on top of the base class.
"""

# TODO : More options through filters :
#        cap on maxSpeed (realMaxSpeed/maxSpeedCap),
#        sensor deprivation
# TODO : unit testing
# TODO : add time

import math
from collections import deque
import numpy as np

import toolbox


# Code convention :
# A for acceleration, V for velocity.
# Left = 0, right = 1

kL, kR = 0, 1
kX, kY = 0, 1

class DualWheel(object):
    """
    A dual wheel robot.

    The robot's wheels are controlled in velocity.
    An order is a tuple of length 4 with :
        1. the target velocity of the left wheel
        2. how long the left wheel maintains the velocity
        3. the target velocity of the right wheel
        4. how long the right wheel maintains the velocity
    Velocity are in rad/s, and duration in s.

    The sensors of the robots return a tuple of length 3:
        1 and 2. the current velocity of the wheels
        3 and 4. the current position x,y of the robot
        5. the current orientation of the robot
    """

    def __init__(self, whlR = 1.0, whlGap = 1.0, maxVrot = 1.0, maxDur = 5.0, dt = 0.01):
        """Initialization
            @param whlR     the size of the wheels
            @param whlGap   the distance between the wheel
            @param maxVrot  maximum physical rotating speed of the wheels,
                            given is in rad/s.
            @param maxDur   maximum duration of a simulation.
            @param dt       delta of time used to update kinematics.
        """
        # Robot interface
        self.m_feats     = (-4, -3, -2, -1)
        self.Sfeats     = (0, 1)
        self.m_bounds    = 2*((-abs(maxVrot), abs(maxVrot)), (0.0, maxDur))
        # parameters
        self.whlR       = whlR    # in m
        self.whlGap     = whlGap  # in m
        self.maxVrot    = maxVrot # in rad/s
        self._dt        = dt      # in s
        self.maxArot    = 2.0*dt  # in rad/s^2
        # state
        self._reset()

    def _reset(self, **kwargs):
        """Put every state variable back to rest state at origin."""
        self._pos   = [0.0, 0.0] # x, y position
        self._theta = 0.0        # orientation in x,y plane in rad
        self._Vrot  = [0.0, 0.0] # effective rotationnal speed of wheels
                                 # left, then right
        self._tgtVrot = [0.0, 0.0]
        # three last (pos, theta) for acceleration, velocity computation
        self._past  = deque(((0.0, 0.0, 0.0) for _ in xrange(4)))
        self._date = 0
        self.max_span = 0.0

    def __repr__(self):
        return "DualWheel"

    def execute_order(self, order):
        """Execute the order"""
        self._reset()
        self._tgtVrot = [order[0], order[2]]
        self._dur     =  order[1], order[3]
        self._maxdur  = max(order[1], order[3])
        while self._date < self._maxdur:
            self._update_motors()
            self._update_kinematics()
            self._date += self._dt
        return pandas.Series(self._Vrot+self._pos+[self._theta], index = self.Sfeats)

    def _update_motors(self):
        """Update the velocity of the wheels"""
        for side in xrange(2):
            if self._date >= self._dur[side]:
                self._tgtVrot[side] = 0
            if not self._Vrot[side] == self._tgtVrot[side]:
                V_legal = toolbox.clip(self._tgtVrot[side], self.maxVrot)
                self._Vrot[side] = V_legal

    def _update_kinematics(self):
        """Update the state of the robot given current wheel velocities."""
        assert len(self._pos) == 2
        Vl = 2*math.pi*self.whlR*self._Vrot[kL] # in m/s
        Vr = 2*math.pi*self.whlR*self._Vrot[kR]
        l  = self.whlGap
        theta = self._theta
        dt    = self._dt
        dx, dy, dtheta = 0.0, 0.0, 0.0
        if Vr == Vl:
            R = None # infinite
            w = 0
            dx, dy = Vr*dt*math.cos(theta), Vr*dt*math.sin(theta)
            dtheta = 0
        else:
            R = l/2*(Vl+Vr)/(Vr-Vl)
            w = (Vr-Vl)/l
            # ICC = instant center of curvature
            icc_x, icc_y = -R*math.sin(theta), R*math.cos(theta)
            Mrot = np.matrix([[math.cos(w*dt), -math.sin(w*dt), 0.0],
                              [math.sin(w*dt),  math.cos(w*dt), 0.0],
                              [           0.0,             0.0, 1.0],
                             ])
            state = (  Mrot*np.matrix([[-icc_x], [-icc_y], [0.0]])
                     + np.matrix([[icc_x], [icc_y], [w*dt]])
                    )
            dx, dy, dtheta = state[0,0], state[1,0], state[2,0]
        self._pos[kX] += dx
        self._pos[kY] += dy
        self._theta += dtheta
        #self._theta = self._theta % (2*math.pi)
        self._past.pop()
        self._past.append((self._pos[kX], self._pos[kY], self._theta))



# class DualG(DualWheel):
#     """Extend the dual wheel robot by providing dual accelerometers.
#         One is placed on the main axis the base of each wheels
#     """
#     def __init__(self, whlR = 1.0, whlGap = 1.0, maxVrot = 1.0, dt = 0.01):
#         DualWheel.__init__(self, whlR, whlGap, maxVrot, dt)
#
#     def sensors(self):
#         """Returns an unlabelled sensor/motor map of the robot, as a couple
#         of tuple of tuple or of integers, representing motor/sensor id.
#         For the sensors, the tuple structure groups similar sensor together.
#         """
#         return DualWheel.sensors(self) + (((5, 6), (7, 8)),)
#
#     def read_sensors(self):
#         """Return the current value of the sensors"""
#         accL, accR = self._accelerometers()
#         return DualWheel.read_sensors(self) + accL + accR
#
#     def _accelerometers(self):
#         """Return the value of past accelerometers"""
#         l  = self.whlGap
#         past_L, past_R = [], [] #
#         for i in xrange(3):
#             x, y, theta = self._past[-(1+i)]
#             pos_l = x - l/2*math.sin(theta), y + l/2*math.cos(theta)
#             pos_r = x - l/2*math.sin(theta), y + l/2*math.cos(theta)
#             past_L.append(pos_l)
#             past_R.append(pos_r)
#         A_l = self._acceleration(past_L)
#         A_r = self._acceleration(past_R)
#         return A_l, A_r
#
#     def _acceleration(self, past):
#         """Compute the immediate acceleration from 3 positions
#             @param list of positions of length at least 3.
#         """
#         dt = self._dt
#         v1x = (past[1][0] - past[2][0])/dt
#         v1y = (past[1][1] - past[2][1])/dt
#         v2x = (past[0][0] - past[1][0])/dt
#         v2y = (past[0][1] - past[1][1])/dt
#         accx = (v2x-v1x)/dt
#         accy = (v2y-v1y)/dt
#         return (accx, accy)
#
#     def _speed(self, past):
#         """Compute the immediate speed from 2 positions
#             @param list of positions of length at least 2.
#         """
#         dt = self._dt
#         v_x = (past[0][0] - past[1][0])/dt
#         v_y = (past[0][1] - past[1][1])/dt
#         return (v_x, v_y)
#
# class Thrower(DualG):
#     """DualG robot that is able to throw balls"""
#
#     def __init__(self, whlR = 1.0, whlGap = 1.0, maxVrot = 1.0, dt = 0.01, armD = 2.0):
#         self.armD = armD # the length of the arm launching the ball.
#                          # as you increase the length, the lauching position is
#                          # decentered and a spinning motion is more and more effective
#         DualG.__init__(self, whlR = whlR, whlGap = whlGap, maxVrot = maxVrot, dt = dt)
#
#     def reset(self, **kwargs):
#         DualG.reset(self)
#         self._launched   = False             # if _launched is True, _ball stores the
#         self._launchdata = None              # date, angle, velocity of launch
#         self._ball       = [0.0, -self.armD] # position of the ball, in absolute coo.
#
#     def sensors(self):
#         # additional sensors for the ball position, expressed in coordinates
#         # relative to the position but not the orientation of the robot.
#         return DualWheel.sensors(self) + ((9, 10),)
#
#     def prims(self):
#         # throwing prim                   id  angle (rad)   velocity (m/s)
#         return DualWheel.prims(self) + ((2, (0.0, 90.0,), (0.0, 1.0)),)
#
#     def read_sensors(self):
#         """Return the current value of the sensors"""
#         return DualG.read_sensors(self) + self._ball_pos()
#
#     def _decrypt_orders(self):
#         """Translate order in target wheel rotation. Prune old orders.
#
#         Only takes care of motor primitive (0, 1, 2), can be safely serialized
#         with other decryptors for other primitive ids.
#         """
#         order_change = self._order_change
#         DualG._decrypt_orders(self)
#         valid_orders = []
#         if order_change:
#             for order in self.orders:
#                 mid, params, onset = order
#                 if mid == 2:
#                     assert len(params) == 2
#                     if onset <= self._date:
#                         # Compute ball new absolute position.
#                         angle, velocity = params
#                         self._launchdata = onset, angle, velocity
#                     if self._date < onset :
#                         valid_orders.append(order)
#                 else:
#                     valid_orders.append(order)
#             self.orders = valid_orders
#         self._order_change = False
#
#     def _update_motors(self):
#         """Update the velocity of the wheels and launch the ball
#         if appropriate.
#         """
#         DualG._update_motors(self)
#         if self._launchdata is not None:
#             launch_date, angle, velocity = self._launchdata
#             if self._date <= launch_date < self._date + self._dt:
#                 self._launch_ball(angle, velocity)
#                 self._launchdata = None
#                 self._launched   = True
#
#     def _ball_pos(self):
#         """Return the ball position, relative to the position of the robot,
#         but not its orientation.
#         """
#         if not self._launched:
#             return (-self.armD*math.cos(self._theta),
#                     -self.armD*math.sin(self._theta))
#         else:
#             return (self._ball[kX] - self._pos[kX],
#                     self._ball[kY] - self._pos[kY])
#
#     def _launch_ball(self, angle, velocity):
#         """"""
#         if not self._launched:
#             dbx, dby = Thrower._projectile_landing(self._theta, angle, velocity)
#             self._ball = (self._pos[kX] + dbx,
#                           self._pos[kY] + dby)
#             self._launched = True
#
#     @staticmethod
#     def _projectile_landing(theta, phi, velocity):
#         """Return the x, y landing position of a projectile
#
#         The starting position is assumed to the plan origin, no friction and the
#         projectile stop on impact at altitude zero.
#         @param theta     the angle in the plan x, y.
#         @param phi       the angle in z.
#         @param velocity  the initial velocity of the projectile in m/s
#         """
#         G = 6.67428*pow(10, -11)
#         t_end = 2*velocity*math.sin(phi)/G
#         d = velocity*math.cos(phi)*t_end
#         return d*math.cos(theta), d*math.sin(theta)
#
#
#
#
#


