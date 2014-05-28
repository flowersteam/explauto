"""
First instanciate another simple arm environment, using the same class and the config you want:
"""

hd_env = env_cls(**env_configs['high_dimensional'])

"""
Then you can plot arm shapes for random motor configurations as above (here 100 samples):
"""

motor_configurations = hd_env.random_motors(n=100)
ax = axes()
for m in motor_configurations:
    hd_env.plot_arm(ax, m)
