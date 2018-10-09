import os
import time

from numpy import array, hstack, float32, zeros, linspace, shape, mean, log2, transpose, sum, isnan
from explauto.utils import bounds_min_max
from explauto.models.dmp import DmpPrimitive
from explauto.environment.environment import Environment
from explauto.models.dmp import DmpPrimitive
from diva_synth import DivaOctaveSynth, DivaMatlabSynth


class DivaEnvironment(Environment):
    def __init__(self, 
                synth, 
                m_mins, m_maxs, s_mins, s_maxs,
                m_used,
                s_used,
                audio,
                diva_path=None):
        
        self.m_mins = m_mins
        self.m_maxs = m_maxs 
        self.s_mins = s_mins
        self.s_maxs = s_maxs
        self.m_used = m_used
        self.s_used = s_used
        self.audio = audio
    
        self.f0 = 1.
        self.pressure = 1.
        self.voicing = 1.
        
        if self.audio:     
            import pyaudio       
            self.pa = pyaudio.PyAudio()
            self.stream = self.pa.open(format=pyaudio.paFloat32,
                                        channels=1,
                                        rate=11025,
                                        output=True)
            
        if synth == "octave":
            self.synth = DivaOctaveSynth(diva_path)
        elif synth == "matlab":
            self.synth = DivaMatlabSynth()
        else:
            raise NotImplementedError

        self.art = array([0.]*10 + [self.f0, self.pressure, self.voicing])   # 13 articulators is a constant from diva_synth.m in the diva source code
        
        Environment.__init__(self, self.m_mins, self.m_maxs, self.s_mins, self.s_maxs)

    def compute_motor_command(self, m):
        return bounds_min_max(m, self.m_mins, self.m_maxs)

    def compute_sensori_effect(self, m):
        if len(array(m).shape) == 1:
            self.art[self.m_used] = m
            self.art_traj = array([self.art]*20).T
            res = self.synth.execute(2.*(self.art.reshape(-1,1)))
            formants = log2(transpose(res[self.s_used]))
            formants[isnan(formants)] = 0.
            return formants[0]
        else:
            self.art_traj = zeros((13, array(m).shape[0]))
            self.art_traj[10, :] = self.f0
            self.art_traj[11, :] = self.pressure
            self.art_traj[12, :] = self.voicing
            self.art_traj[self.m_used,:] = transpose(m)
            
            res = self.synth.execute(2.*(self.art_traj))
            formants = log2(transpose(res[self.s_used,:]))
            formants[isnan(formants)] = 0.
            return formants
        
    def update(self, m, audio=True):
        m_env = self.compute_motor_command(m)
        s = self.compute_sensori_effect(m_env)
        if self.audio and audio and hasattr(self, "art_traj"):
            sound = self.sound_wave(self.art_traj)
            self.stream.write(sound.astype(float32).tostring())
        return s    

    def sound_wave(self, art_traj, power = 2.):
        synth_art = self.art.reshape(1, -1).repeat(len(art_traj.T), axis=0)
        synth_art[:, :] = art_traj.T
        return power * self.synth.sound_wave(synth_art.T)


class DivaDMPEnvironment(Environment):
    def __init__(self, 
                synth,
                m_mins, m_maxs, s_mins, s_maxs,
                m_used,
                s_used,
                n_dmps,
                n_bfs,
                dmp_move_steps,
                dmp_max_param,
                sensory_traj_samples,
                audio,
                diva_path=None):
        
        self.m_mins = m_mins
        self.m_maxs = m_maxs 
        self.s_mins = s_mins
        self.s_maxs = s_maxs
        self.m_used = m_used
        self.s_used = s_used
        self.n_dmps = n_dmps
        self.n_bfs = n_bfs
        self.dmp_move_steps = dmp_move_steps
        self.dmp_max_param = dmp_max_param
        self.samples = array(linspace(-1, self.dmp_move_steps-1, sensory_traj_samples + 1), dtype=int)[1:]
        self.audio = audio
    
        self.f0 = 1.
        self.pressure = 1.
        self.voicing = 1.
        
        if self.audio:     
            import pyaudio       
            self.pa = pyaudio.PyAudio()
            self.stream = self.pa.open(format=pyaudio.paFloat32,
                                        channels=1,
                                        rate=11025,
                                        output=True)
        if synth == "octave":
            self.synth = DivaOctaveSynth(diva_path)
        elif synth == "matlab":
            self.synth = DivaMatlabSynth()
        else:
            raise NotImplementedError

        self.art = array([0.]*10 + [self.f0, self.pressure, self.voicing])   # 13 articulators is a constant from diva_synth.m in the diva source code
        self.max_params = array([1.] * self.n_dmps + [self.dmp_max_param] * self.n_bfs * self.n_dmps + [1.] * self.n_dmps)
        
        self.dmp = DmpPrimitive(dmps=self.n_dmps, bfs=self.n_bfs, timesteps=self.dmp_move_steps)
        Environment.__init__(self, self.m_mins, self.m_maxs, self.s_mins, self.s_maxs)

    def compute_motor_command(self, m):
        return bounds_min_max(m, self.m_mins, self.m_maxs)

    def compute_sensori_effect(self, m):
        m = self.trajectory(m)
        self.art_traj = zeros((13, array(m).shape[0]))
        self.art_traj[10, :] = self.f0
        self.art_traj[11, :] = self.pressure
        self.art_traj[12, :] = self.voicing
        self.art_traj[self.m_used,:] = transpose(m)
        
        res = self.synth.execute(2.*(self.art_traj))
        formants = log2(transpose(res[self.s_used,:]))
        formants[isnan(formants)] = 0.
        self.formants_traj = formants
        return list(formants[self.samples,0]) + list(formants[self.samples,1])
    
    def trajectory(self, m):
        y = self.dmp.trajectory(array(m) * self.max_params)
        if len(y) > self.dmp_move_steps: 
            ls = linspace(0,len(y)-1,self.dmp_move_steps)
            ls = array(ls, dtype='int')
            y = y[ls]
        return y
        
    def update(self, mov, audio=True):
        s = Environment.update(self, mov)
        if self.audio and audio:
            self.play_sound(self.art_traj)
        return s    

    def sound_wave(self, art_traj, power=2.):
        synth_art = self.art.reshape(1, -1).repeat(len(art_traj.T), axis=0)
        synth_art[:, :] = art_traj.T
        return power * self.synth.sound_wave(synth_art.T)

    def play_sound(self, art_traj):
        sound = self.sound_wave(art_traj)
        self.stream.write(sound.astype(float32).tostring())
        
