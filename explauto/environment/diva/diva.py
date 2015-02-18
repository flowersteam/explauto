import os
import pymatlab
from numpy import array, hstack

from ..environment import Environment
from ...utils import bounds_min_max

class DivaSynth:
    def __init__(self, sample_rate=11025):
        self.session = pymatlab.session_factory()
        # sample rate setting not working yet
        self.session.putvalue('sr', array([sample_rate]))

    def execute(self, art):
        self.session.putvalue('art', art)
        self.session.run('[aud, som, outline] = diva_synth(art, \'audsom\')')
        self.aud = self.session.getvalue('aud')
        self.som = self.session.getvalue('som')
        self.vt = self.session.getvalue('outline')
        return self.aud, self.som, self.vt

    def sound_wave(self, art):
        self.session.putvalue('art', art)
        self.session.run('wave = diva_synth(art, \'sound\')')
        return self.session.getvalue('wave')

    def stop(self):
        del self.session


class DivaEnvironment(Environment):

    def __init__(self, m_mins, m_maxs, s_mins, s_maxs, m_used = None, s_used = None, m_default = None):
        Environment.__init__(self, m_mins, m_maxs, s_mins, s_maxs)
        self.synth = DivaSynth()
        self.m_default = m_default
        if m_default is None:
            self.m_default = array([0.] * 10 + [0.7] * 3) #  Neutral position with phonation
        self.m_used = m_used
        if m_used is None:
            self.m_used = range(13)
        self.s_used = s_used
        if s_used is None:
            self.s_used = range(4)
        self.art = self.m_default   # 13 articulators is a constant from diva_synth.m in the diva source code

    def compute_motor_command(self, m_ag):
        return bounds_min_max(m_ag, self.conf.m_mins, self.conf.m_maxs)

    def compute_sensori_effect(self, m_env):
        self.art[self.m_used] = m_env
        res = self.synth.execute(self.art.reshape(-1,1))[0]
        return res[self.s_used]

    def sound_wave(self, art_traj):
        synth_art = self.m_default.reshape(1, -1).repeat(len(art_traj), axis=0)
        synth_art[:, self.m_used] = art_traj
        return self.synth.sound_wave(synth_art.T)
