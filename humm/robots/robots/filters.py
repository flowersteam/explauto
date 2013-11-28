import importlib
import treedict


defaultcfg = treedict.TreeDict()

defaultcfg.filters.s_feats = None
defaultcfg.filters.s_feats_desc = "The mask of the features who want to get back from the sensors"

defaultcfg.filters.s_bounds_factor = None
defaultcfg.filters.s_bounds_factor_desc = "The factor to multiply the s_bounds of the robots, dimension-wise"

defaultcfg.filters.uniformize = False

defaultcfg.verbose = False

def build_robot(cfg):

    cfg.update(defaultcfg, overwrite = False)
    modpath = cfg.robotclass.split('.')
    rclass = importlib.import_module('.'.join(modpath[:-1]))
    rclass = getattr(rclass, modpath[-1])
    rbot = rclass(cfg)

    if not (cfg.filters.s_feats is None and cfg.filters.s_bounds_factor is None):
        rbot = Filter(rbot, s_feats = cfg.filters.s_feats, s_bounds_factor = cfg.filters.s_bounds_factor)
    if cfg.filters.uniformize:
        rbot = Uniformize(rbot)

    return rbot


class Filter(object):

    def __init__(self, sim, s_feats = None, s_bounds_factor = None):
        self.sim = sim
        self.cfg = sim.cfg

        self.m_feats  = sim.m_feats
        self.s_feats  = tuple(s_feats) if s_feats is not None else sim.s_feats
        self._s_feats_validity = [s_i in self.s_feats for s_i in self.sim.s_feats]
        self.m_bounds = sim.m_bounds

        self.s_bounds_factor = s_bounds_factor
        self.s_bounds = self._compute_s_bounds()

    def _compute_s_bounds(self):

        if not hasattr(self.sim, 's_bounds'):
            return None

        filtered_s_bounds = tuple(b for b, v in zip(self.sim.s_bounds, self._s_feats_validity) if v)
        if self.s_bounds_factor is None:
            return filtered_s_bounds
        else:
            assert len(self.s_feats) == len(self.s_bounds_factor)
            s_bounds = []
            for (s_min, s_max), (f_min, f_max) in zip(filtered_s_bounds, self.s_bounds_factor):
                sf_min = s_max - f_min*(s_max-s_min)
                sf_max = s_min + f_max*(s_max-s_min)
                assert sf_min <= sf_max
                s_bounds.append((sf_min, sf_max))
            return tuple(s_bounds)

    def execute_order(self, order, verbose = False):
        effect = self.sim.execute_order(order, verbose = False)
        effect = self._filtered_effect(effect)
        if verbose or (self.cfg.verbose and not self.cfg.inner_verbose):
            print("{}sim{}: ({}) -> ({}){}".format(gfx.purple, gfx.end,
                                                 ", ".join("{}{:+3.2f}{}".format(gfx.cyan, o_i, gfx.end) for o_i in uni_order),
                                                 ", ".join("{}{:+3.2f}{}".format(gfx.green, e_i, gfx.end) for e_i in uni_effect), '\033[K'))
        return effect

    def _filtered_effect(self, effect):
        return tuple(e_i for e_i, v_i in zip(effect, self._s_feats_validity) if v_i)

    def close(self):
        return self.sim.close()


class Uniformize(object):
    """Uniformize the bounds of the order and effect feature between 0.0 and 1.0"""

    def __init__(self, sim):
        self.sim = sim
        self.cfg = sim.cfg

        self.m_feats = sim.m_feats
        self.s_feats = sim.s_feats

        self.m_bounds = len(sim.m_bounds)*((0.0, 1.0),)
        self.s_bounds = len(sim.s_bounds)*((0.0, 1.0),)

    def _uni2sim(self, order):
        return tuple(e_i*(b_max - b_min) + b_min for e_i, (b_min, b_max) in zip(order, self.sim.m_bounds))

    def _sim2uni(self, effect):
        return tuple((e_i - s_min)/(s_max - s_min) for e_i, (s_min, s_max) in zip(effect, self.sim.s_bounds))

    def execute_order(self, uni_order, verbose = False):
        order = self._uni2sim(uni_order)
        effect = self.sim.execute_order(order, verbose = False)
        uni_effect = self._sim2uni(effect)
        if verbose or (self.cfg.verbose and not self.cfg.inner_verbose):
            print("{}sim{}: ({}) -> ({}){}".format(gfx.purple, gfx.end,
                                                 ", ".join("{}{:+3.2f}{}".format(gfx.cyan, o_i, gfx.end) for o_i in uni_order),
                                                 ", ".join("{}{:+3.2f}{}".format(gfx.green, e_i, gfx.end) for e_i in uni_effect), '\033[K'))

        return uni_effect

    def close(self):
        return self.sim.close()
