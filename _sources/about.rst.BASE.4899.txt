About Test
=========

Exploration strategies in Developmental Robotics
------------------------------------------------

An important challenge in Developmental Robotics is how robots can efficiently learn sensorimotor mappings by experience, i.e. the mappings between the motor actions they make and the sensory effects they produce. This can be a robot learning how arm movements make physical objects move, or how movements of a virtual vocal tract modulates vocalization sounds.

Learning such mappings involves machine learning algorithms, which are typically regression algorithms to learn forward models, from motor controllers to sensory effects, and optimization algorithms to learn inverse models, from sensory effects, or goals, to the motor programs allowing to reach them.

The problem is that, for most robotic systems, these spaces are high dimensional, the mapping between them is non-linear and redundant, and there is limited time allowed for learning. Thus, if robots explore the world in an unorganized manner, e.g. randomly, learning algorithms will be often ineffective because very sparse data points will be collected. Data are precious due to the high dimensionality and the limited time, whereas date are not equally useful due to non-linearity and redundancy.
This is why learning has to be guided using efficient exploration strategies.

In the recent year, work in developmental learning has explored in particular two families of algorithmic principles which allow the efficient guiding of learning and exploration.

First, the principle of goal babbling was proposed independantly by `Oudeyer and Kaplan in 2008`_ and `Rolf and Steil in 2010`_. It consists in sampling goals in the sensory effect space and to use the current state of an inverse model to infer a motor action supposed to reach the goals. This strategy allows a progressive covering of the reachable sensory space much more uniformly than in a motor babbling strategy, where the agent samples directly in the motor space.

The second principle is that of active learning and intrinsic motivation, where physical experiments are chosen to gather maximal information gain. Efficient versions of such mechanisms are based on the active choice of learning experiments that maximize learning *progress*, for e.g. improvement of predictions or of competences to reach goals (`Schmidhuber, 1991`_ ; `Oudeyer, 2007`_). This automatically drives the system to explore and learn first easy skills, and then explore skills of progressively increasing complexity.


.. _Rolf and Steil in 2010: http://cor-lab.org/system/files/RolfSteilGienger-TAMD2010-GoalBabbling.pdf
.. _Oudeyer and Kaplan in 2008: http://www.pyoudeyer.com/epirob08OudeyerKaplan.pdf
.. _Schmidhuber, 1991: http://web.media.mit.edu/~alockerd/reading/Schmidhuber-curiositysab-1.pdf
.. _Oudeyer, 2007: http://www.pyoudeyer.com/ims.pdf
