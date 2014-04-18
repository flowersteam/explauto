import importlib


sensorimotor_models = {}

for mod_name in ['non_parametric', ]:
    module = importlib.import_module('explauto.sensorimotor_model.{}'.format(mod_name))

    im = getattr(module, 'sensorimotor_model')
    conf = getattr(module, 'configurations')

    sensorimotor_models[mod_name] = (im, conf)
