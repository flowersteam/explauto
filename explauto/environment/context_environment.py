
import numpy as np

from explauto.environment.environment import Environment
from ..utils import bounds_min_max
from ..utils import rand_bounds


class ContextEnvironment(Environment):

    def __init__(self, env_cls, env_conf, context_mode):
        self.rest_position = context_mode['rest_position']
        self.context_mode = context_mode        
        self.env = env_cls(**env_conf)          
        
        Environment.__init__(self, 
                            np.hstack((self.env.conf.m_mins, self.context_mode['dm_bounds'][0])), 
                            np.hstack((self.env.conf.m_maxs, self.context_mode['dm_bounds'][1])),
                            np.hstack((self.env.conf.s_mins, self.context_mode['ds_bounds'][0])),
                            np.hstack((self.env.conf.s_maxs, self.context_mode['ds_bounds'][1])))                                            
        self.reset()
        
    def reset(self):
        self.current_motor_position = np.array(self.rest_position)
        self.current_sensori_position = np.array(self.env.update(self.current_motor_position, reset=True))
    
    def random_dm(self, n=1):
        return rand_bounds(self.conf.bounds[:, len(self.conf.m_dims)/2:len(self.conf.m_dims)], n)
    
    def compute_motor_command(self, ag_state):
        return bounds_min_max(ag_state, self.conf.m_mins, self.conf.m_maxs)
    
    def compute_sensori_effect(self, mdm):
        m = mdm[:len(mdm)/2]
        dm = mdm[len(mdm)/2:]
        if self.context_mode['choose_m'] == True:  
            s = self.env.update(m, reset=False)
            self.current_motor_position = m
            self.current_sensori_position = s
        else:
            s = self.current_sensori_position
        self.current_motor_position = self.env.compute_motor_command(self.current_motor_position + np.array(dm))
        new_sensori_position = np.array(self.env.update(self.current_motor_position, reset=False))
        ds = new_sensori_position - self.current_sensori_position
        self.current_sensori_position = new_sensori_position
        #print "Context environment, m", m, "dm", dm, "s", s, "ds", ds
        return np.hstack((s, ds))
    
    def plot(self, ax, **kwargs):
        self.env.plot(ax, self.current_motor_position, self.current_sensori_position, **kwargs)