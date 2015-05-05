import os
from numpy import array, hstack, float32, zeros, linspace, shape, mean

from oct2py import octave
from ..environment import Environment
from ...utils import bounds_min_max
from ...models.dmp import DmpPrimitive


if not (os.environ.has_key('AVAKAS') and os.environ['AVAKAS']):
    import pyaudio

                       
                       
class DivaSynth:
    def __init__(self, sample_rate=11025):
        # sample rate setting not working yet
        diva_path = os.path.join(os.getenv("HOME"), 'software/DIVAsimulink/')
        assert os.path.exists(diva_path)
        octave.addpath(diva_path)

    def execute(self, art):
        self.aud, self.som, self.vt = octave.diva_synth(art, 'audsom')
        return self.aud, self.som, self.vt

    def sound_wave(self, art):
        wave = octave.diva_synth(art, 'sound')
        return wave

    def stop(self):
        del self.session



class DivaEnvironment(Environment):
    def __init__(self, config, **kwargs):
        for key, value in kwargs.iteritems():
            setattr(self, key, value)
        
        self.config = config
        if (os.environ.has_key('AVAKAS') and os.environ['AVAKAS']):
            self.audio = False
        
        if self.audio:            
            self.pa = pyaudio.PyAudio()
            self.stream = self.pa.open(format=pyaudio.paFloat32,
                                        channels=1,
                                        rate=11025,
                                        output=True)
            
        self.synth = DivaSynth()
        self.art = array([0.] * 10 + [0.7] * 3)   # 13 articulators is a constant from diva_synth.m in the diva source code
        
        default = zeros(self.config.n_dmps_diva*(self.config.n_bfs_diva+2))
        if not self.config.diva_use_initial:
            init_position = self.rest_position_diva
            default[:self.config.n_dmps_diva] = init_position
        if not self.config.diva_use_goal:
            end_position = self.rest_position_diva
            default[-self.config.n_dmps_diva:] = end_position
        
        self.dmp = DmpPrimitive(self.config.n_dmps_diva, self.config.n_bfs_diva, self.config.used_diva, default, type='discrete')


    def compute_motor_command(self, m_ag):
        return bounds_min_max(m_ag, self.m_mins, self.m_maxs)


    def compute_sensori_effect(self, m_env):
        #print "compute_se", m_env
        self.art[self.m_used] = m_env
        #print self.art
        res = self.synth.execute(self.art.reshape(-1,1))[0]
        
            
        #print "compute_se result", res[self.s_used].flatten()
        return res[self.s_used].flatten()


    def rest_params(self):
        dims = self.config.n_dmps_diva*self.config.n_bfs_diva
        if self.config.diva_use_initial:
            dims += self.config.n_dmps_diva
        if self.config.diva_use_goal:
            dims += self.config.n_dmps_diva
        rest = zeros(dims)
        if self.config.diva_use_initial:
            rest[:self.config.n_dmps_diva] = self.rest_position_diva
        if self.config.diva_use_goal:
            rest[-self.config.n_dmps_diva:] = self.rest_position_diva
        return rest
    
    
    def trajectory(self, m):
        y = self.dmp.trajectory(m)
        if len(y) > self.config.move_steps: 
            ls = linspace(0,len(y)-1,self.config.move_steps)
            ls = array(ls, dtype='int')
            y = y[ls]

        #print "DMP agent. size : ", len(y), y
        #print "goal of dmp : ", self.m[-self.config.n_dmps:]
        
        #print "diva traj", m, y
        y = self.compute_motor_command(y) # check motor bounds
        
        #print "diva traj", m, y
        return y
        
        
    def update(self, mov, log):
        s = Environment.update(self, mov, log)
        
        if self.audio:         
            sound = self.sound_wave(s)
            self.stream.write(sound.astype(float32).tostring())
            
        return list(mean(array(s), axis=0))
        
        
    def sound_wave(self, art_traj):
        synth_art = self.art.reshape(1, -1).repeat(len(art_traj), axis=0)
        #print shape(synth_art), self.m_used, art_traj
        #synth_art[:, self.m_used] = art_traj
        #print "sound wave", self.synth.sound_wave(synth_art.T)
        return 10.*self.synth.sound_wave(synth_art.T)
