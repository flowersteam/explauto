import importlib


sensorimotor_models = {}

for mod_name in ['non_parametric', 'nearest_neighbor', 'ilo_gmm']:
    module = importlib.import_module('explauto.sensorimotor_model.{}'.format(mod_name))

    models = getattr(module, 'sensorimotor_models')

    for name, (sm, conf) in models.iteritems():
        sensorimotor_models[name] = (sm, conf)
