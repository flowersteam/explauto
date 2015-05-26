import os
from numpy import array, hstack, float32, zeros, linspace, shape, mean, log2, transpose, sum, isnan

from oct2py import Oct2Py, Oct2PyError
from ..environment import Environment
from ...utils import bounds_min_max
from ...models.dmp import DmpPrimitive
from invest import ART_DATA_DIR


if not (os.environ.has_key('AVAKAS') and os.environ['AVAKAS']):
    import pyaudio

                       
                       
class DivaSynth:
    def __init__(self, sample_rate=11025):
        # sample rate setting not working yet
        self.diva_path = os.path.join(os.getenv("HOME"), 'software/DIVAsimulink/')
        assert os.path.exists(self.diva_path)
        self.octave = Oct2Py()
        self.restart_iter = 500
        self.init_oct()

    def init_oct(self):
        self.octave.addpath(self.diva_path)
        self.iter = 0
        
    def execute(self, art):
        try:
            self.aud, self.som, self.vt = self.octave.diva_synth(art, 'audsom')
        except Oct2PyError:
            self.reboot()
            print "Warning: Oct2Py crashed, Oct2Py restarted"
            self.aud, self.som, self.vt = self.octave.diva_synth(art, 'audsom')
        self.add_iter()
        return self.aud, self.som, self.vt

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
        self.octave = Oct2Py()
        self.init_oct()
        
    def restart(self):
        self.octave.restart()
        self.init_oct()
        
    def stop(self):
        self.octave.exit()



class DivaEnvironment(Environment):
    def __init__(self, **kwargs):
        for key, value in kwargs.iteritems():
            setattr(self, key, value)
        
        if (os.environ.has_key('AVAKAS') and os.environ['AVAKAS']):
            self.audio = False
        
        if self.audio:            
            self.pa = pyaudio.PyAudio()
            self.stream = self.pa.open(format=pyaudio.paFloat32,
                                        channels=1,
                                        rate=11025,
                                        output=True)
            
        self.synth = DivaSynth()
        self.art = array([0.]*10 + [1]*3)   # 13 articulators is a constant from diva_synth.m in the diva source code
        
        dmp_default = zeros(self.n_dmps_diva*(self.n_bfs_diva+2))
        if not self.diva_use_initial:
            init_position = self.rest_position_diva
            dmp_default[:self.n_dmps_diva] = init_position
        if not self.diva_use_goal:
            end_position = self.rest_position_diva
            dmp_default[-self.n_dmps_diva:] = end_position
        
        self.dmp = DmpPrimitive(self.n_dmps_diva, self.n_bfs_diva, self.used_diva, dmp_default, type='discrete')
        
        self.default_m = zeros(self.n_dmps_diva * self.n_bfs_diva + self.n_dmps_diva * self.diva_use_initial + self.n_dmps_diva * self.diva_use_goal)
        self.default_m_traj = self.compute_motor_command(self.default_m)
        self.default_sound = self.synth.execute(self.art.reshape(-1,1))[0]
        self.default_formants = None
        self.default_formants = self.compute_sensori_effect(self.default_m_traj)
        
        Environment.__init__(self, self.m_mins, self.m_maxs, self.s_mins, self.s_maxs)

    def compute_motor_command(self, m_ag):
        return bounds_min_max(self.trajectory(m_ag), self.m_mins, self.m_maxs)


    def compute_sensori_effect(self, m_env):
        #print "compute_se", m_env
        if len(array(m_env).shape) == 1:
            self.art[self.m_used] = m_env
            #print self.art
            
            if m_env == self.default_m:
                res = self.default_sound
            else:
                res = self.synth.execute(self.art.reshape(-1,1))[0]
            #print "compute_se result", res[self.s_used].flatten()
            formants = log2(transpose(res[self.s_used]))
            formants[isnan(formants)] = 0.
            return formants
        else:
            
            if self.default_formants is not None and (m_env == self.default_m_traj).all():
                return self.default_formants
            else:
                self.art_traj = zeros((13, array(m_env).shape[0]))
                self.art_traj[11:13, :] = 1
                self.art_traj[self.m_used,:] = transpose(m_env)
                res = self.synth.execute(self.art_traj)[0]
                if isnan(sum(log2(transpose(res[self.s_used,:])))):
                    print "diva NaN:"
    #                 print "m_env", m_env
    #                 print "self.art_traj", self.art_traj, 
    #                 print "res", res, 
    #                 print "formants", log2(transpose(res[self.s_used,:]))
                formants = log2(transpose(res[self.s_used,:]))
                formants[isnan(formants)] = 0.
                return formants

    def rest_params(self):
        dims = self.n_dmps_diva*self.n_bfs_diva
        if self.diva_use_initial:
            dims += self.n_dmps_diva
        if self.diva_use_goal:
            dims += self.n_dmps_diva
        rest = zeros(dims)
        if self.diva_use_initial:
            rest[:self.n_dmps_diva] = self.rest_position_diva
        if self.diva_use_goal:
            rest[-self.n_dmps_diva:] = self.rest_position_diva
        return rest
    
    
    def trajectory(self, m):
        y = self.dmp.trajectory(m)
        if len(y) > self.move_steps: 
            ls = linspace(0,len(y)-1,self.move_steps)
            ls = array(ls, dtype='int')
            y = y[ls]

        return y
        
        
    def update(self, mov):
        s = Environment.update(self, mov, batch=True)
        
        if self.audio:         
            sound = self.sound_wave(s)
            self.stream.write(sound.astype(float32).tostring())
            print "Sound sent"
            
        if len(shape(array(s))) == 1:
            return s
        else:
            s = list(mean(array(s), axis=0))
            #print "Diva s=", s
            return s
        
        
    def sound_wave(self, art_traj, power = 4.):
        synth_art = self.art.reshape(1, -1).repeat(len(art_traj), axis=0)
        #print shape(synth_art), self.m_used, art_traj
        #synth_art[:, self.m_used] = art_traj
        #print "sound wave", self.synth.sound_wave(synth_art.T)
        return power * self.synth.sound_wave(synth_art.T)
