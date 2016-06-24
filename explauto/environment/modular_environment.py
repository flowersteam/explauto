import numpy as np

from .environment import Environment


class FlatEnvironment(Environment):
    """
        Combine several environments in parallel with no hierarchy
        
        :param class list envs_cls: the list of environment classes to be combined

        :param dict list envs_cfg: the list of environment configurations

        :param function combined_s: how to combine the sensory outputs of environments
                                    takes the list of all outputs as argument
                                    
    """
    def __init__(self, s_mins, s_maxs, envs_cls, envs_cfg, combined_s):
        
        self.envs = [cls(**cfg) for cls,cfg in zip(envs_cls, envs_cfg)]
        self.n_envs = len(self.envs)
        self.n_params_envs = [len(env.conf.m_dims) for env in self.envs]
        self.n_params_envs_cumsum = np.cumsum([0] + self.n_params_envs)
        self.combined_s = combined_s
    
        config = dict(m_mins=[dim for env in self.envs for dim in env.conf.m_mins],
                      m_maxs=[dim for env in self.envs for dim in env.conf.m_maxs],
                      s_mins=s_mins,
                      s_maxs=s_maxs)
        
        Environment.__init__(self, **config)
        
        
    def reset(self):
        for env in self.envs:
            env.reset()
        
    def rest_position(self):
        return [dim for env in self.envs for dim in env.rest_position]
        
    def rest_params(self):
        return [dim for env in self.envs for dim in env.rest_params()]
        
    def get_m_env(self, m, i):
        m = np.array(m)
        dim_beg = self.n_params_envs_cumsum[i]
        if len(m.shape) == 1:
            return m[range(dim_beg,dim_beg + self.n_params_envs[i])]
        else:
            return m[:,range(dim_beg,dim_beg + self.n_params_envs[i])]
    
    def compute_motor_command(self, m):
        assert len(m) == self.conf.m_ndims, ((self.envs,m,self.conf.m_ndims))
        return m
    
    def compute_sensori_effect(self, m):
        if len(np.array(m).shape) == 1:
            result = self.combined_s([si for i,env in zip(range(self.n_envs), self.envs) for si in list(env.update(self.get_m_env(m, i), reset=False, log=False))])
        else:
            results_envs = [list(env.update(self.get_m_env(m, i), reset=False, log=False)) for i,env in zip(range(self.n_envs), self.envs)]
            result = []
            for i in range(len(m)):   
                result.append(self.combined_s([si for env_results in results_envs for si in env_results[i]]))
        assert len(result) == self.conf.s_ndims
        return result
    
    def plot(self, ax, i, **kwargs_plot):
        for env in self.envs:
            env.plot(ax, i, **kwargs_plot)
            
    def plot_update(self, ax, i, **kwargs_plot):
        lines = []
        for env in self.envs:
            lines += env.plot_update(ax, i, **kwargs_plot)
        return lines
        
            
    
class HierarchicalEnvironment(Environment):
    """
        Combine two environments in hierarchy
        
        :param class top_env_cls: class of top environment

        :param class lower_env_cls: class of lower environment

        :param dict top_env_cfg: configuration of top environment 

        :param dict lower_env_cfg: configuration of lower environment 

        :param function fun_m_lower: which input variables to give to the lower environment  
                                     takes the global input m as argument

        :param function fun_s_lower: which input variables to give to the top environment  
                                     takes the global input m and the output of lower environment as argument

        :param function fun_s_top: which variables to output: takes the global input m, 
                                the output of lower environment and the output of the top environment as argument
                                    
    """
    def __init__(self, m_mins, m_maxs, s_mins, s_maxs, top_env_cls, lower_env_cls, top_env_cfg, lower_env_cfg, fun_m_lower, fun_s_lower, fun_s_top):
        
        self.top_env = top_env_cls(**top_env_cfg)
        self.lower_env = lower_env_cls(**lower_env_cfg)

        self.n_params_lower_env = len(self.lower_env.conf.m_dims)

        self.fun_m_lower = fun_m_lower
        self.fun_s_lower = fun_s_lower
        self.fun_s_top = fun_s_top
        
        config = dict(m_mins=m_mins,
                      m_maxs=m_maxs,
                      s_mins=s_mins,
                      s_maxs=s_maxs)
        
        Environment.__init__(self, **config)
        
    def reset(self):
        self.top_env.reset()
        self.lower_env.reset()
    
    def rest_position(self):
        return self.lower_env.rest_position
        
    def rest_params(self):
        return self.lower_env.rest_params()
    
    def compute_motor_command(self, m):
        assert len(m) == self.conf.m_ndims
        return m
    
    def compute_sensori_effect(self, m):
        if len(np.array(m).shape) == 1:
            s_lower = list(self.lower_env.update(self.fun_m_lower(m), reset=False, log=False))
            s_lower_upd = self.fun_s_lower(m, s_lower)
        else:
            results_lower = list(self.lower_env.update(self.fun_m_lower(m), reset=False, log=False))
            s_lower = []
            s_lower_upd = []
            if isinstance(results_lower[-1], list):
                for i in range(len(m)):   
                    s_lower.append(results_lower[i])
                    s_lower_upd.append(self.fun_s_lower(m, s_lower[-1]))
            else: # if lower env take a trajectory as input but output only one state
                s_lower = results_lower
                s_lower_upd = self.fun_s_lower(m, s_lower)
        top_upd = list(self.top_env.update(s_lower_upd, reset=False, log=False))
        
        s = self.fun_s_top(m, s_lower, top_upd)
        
        assert len(s) == self.conf.s_ndims
        return s
    
    def plot(self, ax, i, **kwargs_plot):
        self.lower_env.plot(ax, i, **kwargs_plot)
        self.top_env.plot(ax, i, **kwargs_plot)
        
    def plot_update(self, ax, i, **kwargs_plot):
        lower_lines = self.lower_env.plot_update(ax, i, **kwargs_plot)
        top_lines = self.top_env.plot_update(ax, i, **kwargs_plot)
        return lower_lines + top_lines
        
