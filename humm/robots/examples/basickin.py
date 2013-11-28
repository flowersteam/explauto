import robots

print 'Loading 6 DOFs kinematic arm...'
robotArm = robots.KinematicArm2D(dim=6)
print '  m_feats  : %s' % (robotArm.m_feats,)
print '  m_bounds : %s' % (robotArm.m_bounds,)
print '  s_feats  : %s' % (robotArm.s_feats,)
print ''

print 'Executing orders...'
order = tuple(0.0 for _ in range(6))
print '  order : %s\t result : %s' % (order, robotArm.execute_order(order))
order = tuple(10.0 for _ in range(6))
print '  order : %s\t result : %s' % (order, robotArm.execute_order(order))
