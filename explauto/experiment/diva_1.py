import sys
from numpy import array, zeros, ones, hstack, maximum, minimum 
from numpy.random import randn

sys.path.append('../model/')
sys.path.append('../environment/diva/')
sys.path.append('../imleSource/python/')

import diva
diva_path = '/home/clement/Documents/Boulot/INRIA_FLOWERS/CODE/DIVAsimulink'

m_dims = range(7)
s_dims = range(7, 9)
i_dims = s_dims
inf_dims = m_dims

m_ndims = len(m_dims)
s_ndims = len(s_dims)

ms_bounds=array([[-1], [1]]).dot(ones((1, m_ndims))) 
ms_bounds = hstack((ms_bounds, array([[100., 500.], [700., 2500.]]) ) )
#ms_bounds[0, -2] = -0.5
noise_env = 200.
sigma0_sm = 0.2 ** 2
psi0_sm = [noise_env ** 2] * s_ndims

sigma0_i = (200.) **2
psi0_i = [0.2 ** 2]

class Env:
    def __init__(self, diva_path):
        self.synth = diva.DivaSynth(diva_path)
    def execute(self,m):
        m = maximum(m.flatten(), ms_bounds[0:1,m_dims])
        m = minimum(m.flatten(), ms_bounds[1:2,m_dims])
        s = self.synth.execute(hstack((m, zeros((1, 13 - m_ndims)))).reshape(-1, 1))[1:3]
        s=array(s).reshape(-1,1)
        s += noise_env*randn(len(s_dims),1)
        return m.reshape(-1,1),s

env=Env(diva_path)

import exploration
from competence import competence_1 as competence
import sm_model as smm
import i_model as im

sm_model = smm.ImleModel(m_dims, s_dims, sigma0_sm, psi0_sm)
i_model = im.RandomInterest(ms_bounds[:,i_dims])
#i_model = im.ProgressInterest(ms_bounds[:,i_dims], sigma0_i, psi0_i)
#i_model = im.GmmInterest(ms_bounds[:,i_dims], sigma0_i, psi0_i)

ag=exploration.Agent(m_dims=m_dims, s_dims=s_dims, i_dims=i_dims, inf_dims=inf_dims, env=env, sm_model=sm_model, i_model=i_model, competence=competence)
