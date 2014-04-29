#!/usr/bin/python
# -*- coding: utf-8 -*-

########################################################################
#  File Name	: 'generate_simple_lip.py'
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

from numpy import *
from scipy import *
import pylab as P

import random as rd

import simple_lip as lip


XMIN = -pi
XMAX = pi

VMIN = -2.5
VMAX = 2.5

UMIN = -0.25
UMAX = 0.25


def generate(XRANGE, DXRANGE, URANGE):
    xt = rd.uniform(XRANGE[0], XRANGE[1])
    dxt = rd.uniform(DXRANGE[0], DXRANGE[1])
    ut = rd.uniform(URANGE[0], URANGE[1])
    res = lip.simulate([xt, dxt], [ut])
    xt1 = res[0]
    dxt1 = res[1]
    return [xt, dxt, ut, xt1, dxt1]


def build(XRANGE, DXRANGE, URANGE, NB=10):
    res = [(generate(XRANGE, DXRANGE, URANGE)) for i in range(NB)]
    return array(res)

rd.seed()


# Ça devrait être assez explicite...
res = build([XMIN, XMAX], [VMIN, VMAX], [UMIN, UMAX], 100)
print res
