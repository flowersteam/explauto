import treedict
import robots
import models.testbed.testcase as testcase

print 'Loading 6 DOFs kinematic arm...'
robotArm = robots.KinematicArm2D(dim=6)

testcasesM = testcase.uniform_motor_testcases(robotArm, n = 100)
print 'Generated %i motor-uniform testcase' % (len(testcasesM),)

testcasesS = testcase.uniform_sensor_testcases(robotArm, n = 100)
print 'Generated %i sensor-uniform testcase' % (len(testcasesS),)
