import pandas

import robots

def f(x):
    return [3*x[0], x[0]*x[1], x[0] + x[1]**2]

r = robots.RobotFunction(f, 2, 3)

y = list(r.execute_order(pandas.Series([0, 0], index = r.m_feats)))
assert y[0] == y[1] == y[2] == 0
