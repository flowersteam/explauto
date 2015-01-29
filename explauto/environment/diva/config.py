from numpy import array
from collections import namedtuple


# Useless names strings, just for information:
articulator_names = ['art' + str(n) for n in range(10)] + ['pitch', 'pressure', 'voicing']
somato_names = ['pharyngeal', 'uvular', 'velar', 'palatal', 'alveodental', 'labial', 'pressure', 'voicing']
auditory_names =['F0', 'F1', 'F2', 'F3']



s_mins = array([0., 0., 500., 2000.])
s_maxs = array([200., 1000., 3000., 4000])


def make_diva_config(m_max, m_used, s_used):
    return dict(m_mins=array([-m_max] * len(m_used)),
                m_maxs=array([m_max] * len(m_used)),
                s_mins=s_mins[s_used],
                s_maxs=s_maxs[s_used],
                m_used=m_used,
                s_used=s_used)


default_config = make_diva_config(1., range(10), range(1, 4))  # Art1-Art10, F1-F3
vowel_config = make_diva_config(1., range(7), range(1, 3))  # Art1-Art7, F1-F2
low_config = make_diva_config(1., range(3), range(1, 3))
full_config = make_diva_config(1., range(13), range(4))  # Art1-Art10-F-P-V, F0-F1-F2-F3
