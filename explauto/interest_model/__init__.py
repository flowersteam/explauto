import importlib


interest_models = {}

for mod_name in ['random', 'learning_progress']:
    module = importlib.import_module('explauto.interest_model.{}'.format(mod_name))

    im = getattr(module, 'interest_model')
    conf = getattr(module, 'configurations')

    interest_models[mod_name] = (im, conf)
