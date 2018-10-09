from __future__ import print_function

import os

from numpy import array, hstack, float32, zeros, linspace, shape, mean, log2, transpose, sum, isnan

from ..environment import Environment
from ...utils import bounds_min_max


class DivaMatlabSynth(object):
    def __init__(self, sample_rate=11025):
        import pymatlab
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

                       
class DivaOctaveSynth(object):
    def __init__(self, diva_path=None):
        import oct2py
        self.oct2py = oct2py
        self.diva_path = diva_path or os.path.join(os.getenv("HOME"), 'software/DIVAsimulink/')
        assert os.path.exists(self.diva_path)
        self.octave = self.oct2py.Oct2Py()
        self.restart_iter = 500
        self.init_oct()

    def init_oct(self):
        self.octave.addpath(self.diva_path)
        self.iter = 0
        
    def execute(self, art):
        try:
            self.aud = self.octave.diva_synth(art, 'audsom')
        except:
            self.reboot()
            print("Warning: Oct2Py crashed, Oct2Py restarted")
            self.aud = self.octave.diva_synth(art, 'audsom')
        self.add_iter()
        return self.aud

    def sound_wave(self, art):
        wave = self.octave.diva_synth(art, 'sound')
        self.add_iter()
        return wave
    
    def add_iter(self):
        if self.iter >= self.restart_iter:
            self.restart()
        else:
            self.iter += 1
            
    def reboot(self):
        self.octave = self.oct2py.Oct2Py()
        self.init_oct()
        
    def restart(self):
        self.octave.restart()
        self.init_oct()
        
    def stop(self):
        self.octave.exit()

