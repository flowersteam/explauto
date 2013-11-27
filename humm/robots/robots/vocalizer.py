"""Implementation of a model of vowel generation and perception by de Boer:
de Boer, B. "The Origin of Vowel Systems", Oxford: Oxford University Press

Formant equations :

F_1 = (  p**2 * (( -392 +  392*r)*h**2 + (  596 -  668*r)*h + ( -146 +  166*r))
       + p    * ((  348 -  348*r)*h**2 + ( -494 -  606*r)*h + (  141 -  175*r))
       +        ((  340 -   72*r)*h**2 + ( -796 +  108*r)*h + (  708 -   38*r)))

F_2 = (  p**2 * ((-1200 + 1208*r)*h**2 + ( 1320 - 1328*r)*h + (  118 -  158*r))
       + p    * (( 1864 - 1488*r)*h**2 + (-2644 + 1510*r)*h + ( -561 +  221*r))
       +        (( -670 +  490*r)*h**2 + ( 1355 -  697*r)*h + ( 1517 -  117*r)))

F_3 = (  p**2 * ((  604 -  604*r)*h**2 + ( 1038 - 1178*r)*h + (  246 +  566*r))
       + p    * ((-1150 + 1262*r)*h**2 + (-1443 + 1313*r)*h + ( -317 -  483*r))
       +        (( 1130 -  836*r)*h**2 + ( -315 +   44*r)*h + ( 2427 -  127*r)))

F_4 = (  p**2 * ((-1120 +   16*r)*h**2 + ( 1696 -  180*r)*h + (  500 +  522*r))
       + p    * (( -140 +  240*r)*h**2 + ( -578 +  214*r)*h + ( -692 -  419*r))
       +        (( 1480 -  602*r)*h**2 + (-1220 +  289*r)*h + ( 3678 -  178*r)))

Cochlea model :

C_1 = F_1
C_2 = | F_2                          if F_3 - F_2 >  c
      | ((2 - w_1)*F_2 + w_1*F_3)/2  if F_3 - F_2 <= c and F_4 - F_2 >= c
      | (w_2*F_2 + (2 - w_2)*F_3)/2  if F_4 - F_2 <= c and F_3 - F_2 <= F_4 - F_3
      | ((2 + w_2)*F_3 - w_2*F_4)/2  if F_4 - F_2 <= c and F_3 - F_2 >= F_4 - F_3

with w_1 = (c - (F_3 - F_2))/2
     w_2 = ((F_4 - F_3) - (F_3 - F_2))/(F_4 - F_2)
"""

import math
import time
#import pandas


class Vocalizer(object):

    def __init__(self, **kwargs):
        """Initialization"""
        self.s_feats = (0, 1)
        self.m_feats = (-3, -2, -1)
        self.m_bounds = 3*((0.0, 1.0),)

    def _legal_order(self, order):
        """Return True if an orders parameters are between bounds"""
        eps = 1e-7
        return all(lb-eps <= p <= ub+eps for p, (lb, ub) in zip(order, self.m_bounds))

    def __repr__(self):
        return "VowelModel"

    def execute_order(self, order):
        """Execute a new order."""
        assert self._legal_order(order), "The order was out of bounds or did not present the correct features."
        r, h, p = order
        self._pos = self._perceive(self._vocalize((r, h, p)))
        self._pos = (self._pos[0]/100, self._pos[1]/100)
        return self._pos

    def _vocalize(self, param):
        r, h, p = param
        F_1 = (  p**2 * (( -392 +  392*r)*h**2 + (  596 -  668*r)*h + ( -146 +  166*r))
               + p    * ((  348 -  348*r)*h**2 + ( -494 +  606*r)*h + (  141 -  175*r))
               +        ((  340 -   72*r)*h**2 + ( -796 +  108*r)*h + (  708 -   38*r)))

        F_2 = (  p**2 * ((-1200 + 1208*r)*h**2 + ( 1320 - 1328*r)*h + (  118 -  158*r))
               + p    * (( 1864 - 1488*r)*h**2 + (-2644 + 1510*r)*h + ( -561 +  221*r))
               +        (( -670 +  490*r)*h**2 + ( 1355 -  697*r)*h + ( 1517 -  117*r)))

        F_3 = (  p**2 * ((  604 -  604*r)*h**2 + ( 1038 - 1178*r)*h + (  246 +  566*r))
               + p    * ((-1150 + 1262*r)*h**2 + (-1443 + 1313*r)*h + ( -317 -  483*r))
               +        (( 1130 -  836*r)*h**2 + ( -315 +   44*r)*h + ( 2427 -  127*r)))

        F_4 = (  p**2 * ((-1120 +   16*r)*h**2 + ( 1696 -  180*r)*h + (  500 +  522*r))
               + p    * (( -140 +  240*r)*h**2 + ( -578 +  214*r)*h + ( -692 -  419*r))
               +        (( 1480 -  602*r)*h**2 + (-1220 +  289*r)*h + ( 3678 -  178*r)))
        return F_1, F_2, F_3, F_4

    def _perceive(self, formants):
        F_1, F_2, F_3, F_4 = formants
        C_1 = F_1
        C_2 = None
        c = 3.5 # Barks
        w_1 = (c - (F_3 - F_2))/c
        w_2 = ((F_4 - F_3) - (F_3 - F_2))/(F_4 - F_2)

        if F_3 - F_2 >  c:
            C_2 = F_2
        elif F_3 - F_2 <= c and F_4 - F_2 >= c:
            C_2 = ((2 - w_1)*F_2 + w_1*F_3)/2
        elif F_4 - F_2 <= c and F_3 - F_2 <= F_4 - F_3:
            C_2 = (w_2*F_2 + (2 - w_2)*F_3)/2
        elif F_4 - F_2 <= c and F_3 - F_2 >= F_4 - F_3:
            C_2 = ((2 + w_2)*F_3 - w_2*F_4)/2
        else:
            assert False
        return C_1/8.0, C_2/16.0

