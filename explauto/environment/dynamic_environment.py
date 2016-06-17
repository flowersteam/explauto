import numpy as np
from matplotlib import animation, rc
from IPython.display import HTML

from explauto.models.dmp import DmpPrimitive
from explauto.utils.utils import bounds_min_max
from explauto.environment.environment import Environment


class DynamicEnvironment(Environment):
    def __init__(self, env_cls, env_cfg, 
                 m_mins, m_maxs, s_mins, s_maxs,
                 n_bfs, move_steps, 
                 n_dynamic_motor_dims, n_dynamic_sensori_dims, max_params,
                 motor_traj_type="DMP", sensori_traj_type="samples", 
                 optim_initial_position=False, optim_end_position=False, default_motor_initial_position=None, default_motor_end_position=None,
                 default_sensori_initial_position=None, default_sensori_end_position=None):
        
        self.env = env_cls(**env_cfg)
        
        self.n_bfs = n_bfs
        self.n_motor_traj_points = self.n_bfs
        self.n_sensori_traj_points = self.n_bfs 
        self.move_steps = move_steps
        self.n_dynamic_motor_dims = n_dynamic_motor_dims
        self.n_dynamic_sensori_dims = n_dynamic_sensori_dims
        self.max_params = max_params
        self.motor_traj_type = motor_traj_type 
        self.sensori_traj_type = sensori_traj_type 
        self.optim_initial_position = optim_initial_position
        self.optim_end_position = optim_end_position
        
        Environment.__init__(self, m_mins, m_maxs, s_mins, s_maxs)
        self.s_traj = None
                
        if self.motor_traj_type == "DMP":
            self.init_motor_DMP(optim_initial_position, optim_end_position, default_motor_initial_position, default_motor_end_position)
        else:
            raise NotImplementedError
            
        if self.sensori_traj_type == "DMP":
            self.init_sensori_DMP(optim_initial_position, optim_end_position, default_sensori_initial_position, default_sensori_end_position)
        elif self.sensori_traj_type == "samples":
            self.samples = np.array(np.linspace(-1, self.move_steps-1, self.n_sensori_traj_points + 1), dtype=int)[1:]
        elif self.sensori_traj_type == "end_point":
            self.end_point = self.move_steps-1
        else:
            raise NotImplementedError
            
    def reset(self):
        self.env.reset()
            
    def init_motor_DMP(self, optim_initial_position=True, optim_end_position=True, default_motor_initial_position=None, default_motor_end_position=None):
        default = np.zeros(self.n_dynamic_motor_dims * (self.n_bfs + 2))
        if not optim_initial_position:
            default[:self.n_dynamic_motor_dims] = default_motor_initial_position or [0.] * self.n_dynamic_motor_dims
            dims_optim = [False] * self.n_dynamic_motor_dims
        else:
            dims_optim = [True] * self.n_dynamic_motor_dims
        dims_optim += [True] * (self.n_dynamic_motor_dims * self.n_bfs)
        if not optim_end_position:
            default[-self.n_dynamic_motor_dims:] = default_motor_end_position or [0.] * self.n_dynamic_motor_dims
            dims_optim += [False] * self.n_dynamic_motor_dims
        else:
            dims_optim += [True] * self.n_dynamic_motor_dims
        self.motor_dmp = DmpPrimitive(self.n_dynamic_motor_dims, 
                                    self.n_bfs, 
                                    dims_optim, 
                                    default, 
                                    type='discrete',
                                    timesteps=self.move_steps)
            
    def init_sensori_DMP(self, optim_initial_position=True, optim_end_position=True, default_sensori_initial_position=None, default_sensori_end_position=None):
        default = np.zeros(self.n_dynamic_sensori_dims * (self.n_sensori_traj_points + 2))
        if not optim_initial_position:
            default[:self.n_dynamic_sensori_dims] = default_sensori_initial_position
            dims_optim = [False] * self.n_dynamic_sensori_dims
        else:
            dims_optim = [True] * self.n_dynamic_sensori_dims
        dims_optim += [True] * (self.n_dynamic_sensori_dims * self.n_sensori_traj_points)
        if not optim_end_position:
            default[-self.n_dynamic_sensori_dims:] = default_sensori_end_position
            dims_optim += [False] * self.n_dynamic_sensori_dims
        else:
            dims_optim += [True] * self.n_dynamic_sensori_dims
        self.sensori_dmp = DmpPrimitive(self.n_dynamic_sensori_dims, 
                                        self.n_sensori_traj_points, 
                                        dims_optim, 
                                        default, 
                                        type='discrete',
                                        timesteps=self.move_steps)
    
    def compute_motor_command(self, m_ag):  
        m_ag = bounds_min_max(m_ag, self.conf.m_mins, self.conf.m_maxs)
        if self.motor_traj_type == "DMP":
            dyn_idx = range(self.n_dynamic_motor_dims * self.n_motor_traj_points)
            m_weighted = m_ag[dyn_idx] * self.max_params
            if self.optim_initial_position:
                m_weighted[:self.n_dynamic_motor_dims] = m_weighted[:self.n_dynamic_motor_dims] / self.max_params
            if self.optim_end_position:
                m_weighted[-self.n_dynamic_motor_dims:] = m_weighted[-self.n_dynamic_motor_dims:] / self.max_params
            m_dyn = self.motor_dmp.trajectory(m_weighted)
            static_idx = range(self.n_dynamic_motor_dims * self.n_motor_traj_points, self.conf.m_ndims)
            m_static = m_ag[static_idx]
            m = [list(m_dyn_param) + list(m_static) for m_dyn_param in list(m_dyn)]
        else:
            raise NotImplementedError
        return m
    
    def compute_sensori_effect(self, m_traj):
        s = self.env.update(m_traj, reset=False, log=False)
        self.s_traj = s
        y = np.array(s[:self.move_steps])
        if self.sensori_traj_type == "DMP":
            self.sensori_dmp.dmp.imitate_path(np.transpose(y))
            w = self.sensori_dmp.dmp.w.flatten()
            s_ag = list(w)    
        elif self.sensori_traj_type == "samples":
            w = y[self.samples,:]
            s_ag = list(np.transpose(w).flatten())
        elif self.sensori_traj_type == "end_point":
            s_ag = y[self.end_point,:].flatten()
        else:
            raise NotImplementedError  
        s = s_ag
        return bounds_min_max(s, self.conf.s_mins, self.conf.s_maxs)    
        
    def update(self, m_ag, reset=True, log=False):
        if reset:
            self.reset()
        if len(np.array(m_ag).shape) == 1:
            s = self.one_update(m_ag, log)
        else:
            s = []
            for m in m_ag:
                s.append(self.one_update(m, log))
            s = np.array(s)
        return s
        
    def plot(self, fig, ax, **kwargs):
        ax.cla()
        ax.set_aspect('equal')
        ax.set_xlim((-1.5, 1.5))
        ax.set_ylim((-1.5, 1.5))
        def animate(i): return tuple(self.env.plot_update(ax, i))
        return animation.FuncAnimation(fig, animate, frames=50, interval=50, blit=True).to_html5_video()



