from pandas import Series

import robots

print 'Loading Ergorobot...'
ergo = robots.Ergorobot()
print '  m_feats  : %s' % (ergo.m_feats,)
print '  m_bounds : %s' % (ergo.m_bounds,)
print '  s_feats  : %s' % (ergo.s_feats,)
print ''

print 'Executing orders...'
order = Series(tuple(0.0 for _ in range(6)), index = ergo.m_feats)
print '  order : %s\t result : %s' % (tuple(order), tuple(ergo.execute_order(order)))
order = Series(tuple(10.0 for _ in range(6)), index = ergo.m_feats)
print '  order : %s\t result : %s' % (tuple(order), tuple(ergo.execute_order(order)))
