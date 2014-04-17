#!/usr/bin/python
# -*- coding: utf-8 -*-

########################################################################
#  File Name	: 'simple_lip.py'
#  Author	: Steve NGUYEN
#  Contact      : steve.nguyen.000@gmail.com
#  Created	: vendredi, janvier 31 2014
#  Revised	:
#  Version	:
#  Target MCU	:
#
#  This code is distributed under the GNU Public License
# 		which can be found at http://www.gnu.org/licenses/gpl.txt
#
#
#  Notes:	notes
########################################################################


from numpy import pi, sin


def simulate(X, U, dt=0.25):
    dtheta = X[1]+(U[0]+sin(X[0]))*dt
    theta = X[0] + X[1]*dt+(dt*dt)/2.0*(U[0]+sin(X[0]))

    if theta > pi:
        theta = theta-2.0*pi
    elif theta < -pi:
        theta = 2.0*pi+theta

    X[0] = theta
    X[1] = dtheta

    return X
