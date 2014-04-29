# To switch from motor to goal babbling
# You just need to change the babling parameter at instanciation

goal_expe = Experiment.from_settings(environment='simple_arm',
                                     babbling='goal',
                                     interest_model='random',
                                     sensorimotor_model='non_parametric')

goal_expe.evaluate_at([1, 10, 20, 30, 100, 200])
goal_expe.bootstrap(5)

goal_expe.run()
