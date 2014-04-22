from collections import namedtuple
from numpy import hstack, array

from ..environment import Environment
from ...utils import bounds_min_max
from .diva import DivaSynth

diva_path = '../../../DIVAsimulink/'

# Useless names strings, just for information:
articulator_names = ['art' + str(n) for n in range(10)] + ['pitch', 'pressure', 'voicing']
somato_names = ['pharyngeal', 'uvular', 'velar', 'palatal', 'alveodental', 'labial', 'pressure', 'voicing']
auditory_names =['F0', 'F1', 'F2', 'F3']

articulator_bounds = tuple((-1., 1.) for _ in range(len(articulator_names)))
somato_bounds = tuple((-1., 1.) for _ in range(len(somato_names)))
auditory_bounds = ((0., 200.), (0., 1000.), (500., 3000.), (2000., 4000.))

bounds = articulator_bounds + auditory_bounds

# high pressure and voicing to ensure phonation
articulator_default = [0.] * 11 + [0.7] * 2

auditory_default = [100., 500., 1500., 3000.]

default = articulator_default + auditory_default

art1__7 = tuple(range(7))  #the 7 principal articulators
F1_F2 = (-3, -2)


def get_config(m_ndims, s_ndims, m_used, s_used):
    return {
            'm_ndims' : m_ndims,
            's_ndims' : s_ndims,
            'm_mins' : array([articulator_bounds[d][0] for d in m_used]),
            'm_maxs' : array([articulator_bounds[d][1] for d in m_used]),
            's_mins' : array([auditory_bounds[d][0] for d in s_used]),
            's_maxs' : array([auditory_bounds[d][1] for d in s_used]),
            'bounds' : tuple([bounds[d] for d in m_used + s_used]),
            'default' :  default,
            'm_used' : m_used,
            's_used' : s_used
            }


class DivaEnvironment(Environment):
    def __init__(self, config_dict):
        Environment.__init__(self, ndims = config_dict['m_ndims'] + config_dict['s_ndims'])
        self.synth = DivaSynth(diva_path)
        for attr in ['m_ndims', 's_ndims', 'bounds', 'default', 'm_used', 's_used']:
            setattr(self, attr, array(config_dict[attr]))
        self.m_mins = array(bounds)[:self.m_ndims,0]
        self.m_maxs = array(bounds)[:self.m_ndims,1]
        self.art = self.default[:13]   # 13 articulators is a constant from diva_synth.m in the diva source code
    def next_state(self, ag_state):
        self.art[self.m_used] = ag_state
        self.art[self.m_used] = bounds_min_max(self.art[self.m_used], self.m_mins, self.m_maxs)
        res = self.synth.execute(self.art.reshape(-1,1))[0]
        self.state[:self.m_ndims] = self.art[self.m_used]
        #self.state[-self.s_ndims:] = hstack((res[0], res[1]))[self.s_used]
        self.state[-self.s_ndims:] = res[self.s_used]
