import importlib
#from .environment import Environment


environments = {}
for mod_name in ['simple_arm', 'pendulum', 'npendulum', 'pypot']:
    try:
        module = importlib.import_module('explauto.environment.{}'.format(mod_name))
        env = getattr(module, 'environment')
        conf = getattr(module, 'configurations')
        testcases = getattr(module, 'testcases')
        environments[mod_name] = (env, conf, testcases)
    except ImportError:
        pass
def available_configurations(environment):
    _, env_configs, _ = environments[environment]
    return env_configs
