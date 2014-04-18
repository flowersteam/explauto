import importlib

from .environment import Environment

environments = {}
for mod_name in ['simple_arm', 'pendulum']:
    module = importlib.import_module('explauto.environment.{}'.format(mod_name))
    env = getattr(module, 'environment')
    conf = getattr(module, 'configurations')
    environments[mod_name] = (env, conf)
