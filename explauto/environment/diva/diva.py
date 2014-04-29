import pymatlab
import numpy as np
import os

class DivaSynth:
    def __init__(self, diva_path, sample_rate = 11025):
        self.session = pymatlab.session_factory()
        self.session.run('path(\'' + diva_path + '\', path)')
        self.session.putvalue('sr', np.array([sample_rate]))
        diva_path = os.path.abspath(diva_path)
        self.session.run(['path(\'', diva_path, '\', path)'])

    def execute(self, art):
        self.session.putvalue('art', art)
        self.session.run('[aud, som, outline] = diva_synth(art, \'audsom\')')
        self.aud = self.session.getvalue('aud')
        self.som = self.session.getvalue('som')
        self.vt = self.session.getvalue('outline')
        return self.aud, self.som, self.vt

    def sound_wave(self, art):
        self.session.putvalue('art', art)
        self.session.run('sr = sr(1)')
        print self.session.getvalue('sr')
        self.session.run('wave = diva_synth(art, \'sound\')')
        return self.session.getvalue('wave')
    
    def stop(self):
        del self.session
