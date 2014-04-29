import random
import robots
import models.learner

# Creating a 6DOF robotic arm
arm6DOF = robots.KinematicArm2D(dim=6)
# Creating a learner, regrouping a forward and inverse model
learner = models.learner.Learner.from_robot(arm6DOF, fwd = 'LWLR', inv = 'L-BFGS-B')

# Training the learner on 1000 examples
for i in range(1000):
    order = [random.uniform(-90, 90) for _ in range(6)] # random order of dim 6
    effect = arm6DOF.execute_order(order)
    learner.add_xy(order, effect)

# Predicting effects.
print learner.predict_effect((14.0, 20.0, -35.0, 61.0, -11.0, 79.0))

# Infering orders.
print learner.infer_order((30.0, -10.0))

