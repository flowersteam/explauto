import importlib

from .environment import Environment

environments = {}
for mod_name in ['simple_arm', 'pendulum']:
    module = importlib.import_module(mod_name)
    env = getattr(module, 'environment')
    conf = getattr(module, 'configurations')
    environments[module] = (env, conf)


