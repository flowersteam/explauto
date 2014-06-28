# To switch from motor to goal babbling
# You just need to change the babling parameter at instanciation

from explauto.experiment import Experiment, make_settings

s_goal = make_settings(environment='simple_arm',
                       babbling_mode='goal',
                       interest_model='random',
                       sensorimotor_model='nearest_neighbor')

goal_expe = Experiment.from_settings(s_goal)

goal_expe.evaluate_at([1, 10, 20, 30, 100, 200, 300, 400], s_goal.default_testcases)

goal_expe.run()

ax = axes()
data = goal_expe.log.scatter_plot(ax, (('sensori', [0, 1]), ), color='green')
