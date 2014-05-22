import importlib


interest_models = {}

for mod_name in ['random', 'gmm_progress', 'discrete_progress']:
    module = importlib.import_module('explauto.interest_model.{}'.format(mod_name))

    models = getattr(module, 'interest_models')

    for name, (im, conf) in models.iteritems():
        interest_models[name] = (im, conf)
