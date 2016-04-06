
import numpy as np

from numpy import array

from ..exceptions import ExplautoBootstrapError
from .sensorimotor_model import SensorimotorModel
from .learner import Learner
from explauto.utils import bounds_min_max
from explauto.utils import rand_bounds
from explauto.interest_model.competences import competence_dist
from explauto.models.dataset import BufferedDataset



class NonParametric(SensorimotorModel):
    """ This class wraps the non-parametric forward and inverse models implemented by Fabien Benureau, in order to fit into the Explauto framework. 
        Original code available at https://github.com/humm/models
        Adapted by Sebastien Forestier at https://github.com/sebastien-forestier/models
    """
    def __init__(self, conf, sigma_explo_ratio=0.1, fwd='LWLR', inv='L-BFGS-B', **learner_kwargs):

        SensorimotorModel.__init__(self, conf)
        for attr in ['m_ndims', 's_ndims', 'm_dims', 's_dims', 'bounds', 'm_mins', 'm_maxs']:
            setattr(self, attr, getattr(conf, attr))

        self.sigma_expl = (conf.m_maxs - conf.m_mins) * float(sigma_explo_ratio)
        self.mode = 'explore'
        mfeats = tuple(range(self.m_ndims))
        sfeats = tuple(range(-self.s_ndims, 0))
        mbounds = tuple((self.bounds[0, d], self.bounds[1, d]) for d in range(self.m_ndims))

        self.model = Learner(mfeats, sfeats, mbounds, fwd, inv, **learner_kwargs)
        self.t = 0
        self.bootstrapped_s = False

    def infer(self, in_dims, out_dims, x):
        if self.t < max(self.model.imodel.fmodel.k, self.model.imodel.k):
            raise ExplautoBootstrapError
        
        if in_dims == self.m_dims and out_dims == self.s_dims:  # forward
            return array(self.model.predict_effect(tuple(x)))
        
        elif in_dims == self.s_dims and out_dims == self.m_dims:  # inverse
            if not self.bootstrapped_s:
                # If only one distinct point has been observed in the sensory space, then we output a random motor command
                return rand_bounds(np.array([self.m_mins, 
                                             self.m_maxs]))[0]
            else:
                if self.mode == 'explore':
                    self.mean_explore = array(self.model.infer_order(tuple(x)))
                    r = self.mean_explore
                    r[self.sigma_expl > 0] = np.random.normal(r[self.sigma_expl > 0], self.sigma_expl[self.sigma_expl > 0])
                    res = bounds_min_max(r, self.m_mins, self.m_maxs)
                    return res#, self.model.imodel.fmodel.dataset.nn_y(x)[1][0]#, self.model.imodel.fmodel.dataset.get_y(self.model.imodel.fmodel.dataset.nn_y(x)[1][0]), array(self.model.predict_effect(res))
                else:  # exploit'
                    return array(self.model.infer_order(tuple(x)))#, -2         
            
        elif out_dims == self.m_dims[len(self.m_dims)/2:]:  # dm = i(M, S, dS)
            if not self.bootstrapped_s:
                # If only one distinct point has been observed in the sensory space, then we output a random motor command
                return rand_bounds(np.array([self.m_mins[self.m_ndims/2:], self.m_maxs[self.m_ndims/2:]]))[0]
            else:
                assert len(x) == len(in_dims)
                m = x[:self.m_ndims/2]
                s = x[self.m_ndims/2:][:self.s_ndims/2]
                ds = x[self.m_ndims/2:][self.s_ndims/2:]
                self.mean_explore = array(self.model.imodel.infer_dm(m, s, ds))               
                if self.mode == 'explore': 
                    r = np.random.normal(self.mean_explore, self.sigma_expl[out_dims])
                    res = bounds_min_max(r, self.m_mins[out_dims], self.m_maxs[out_dims])                
                    return res       
                else:
                    return self.mean_explore
        else:
            raise NotImplementedError
                                
    def predict_given_context(self, x, c, c_dims):
        return self.model.imodel.fmodel.predict_given_context(x, c, c_dims)

    def update(self, m, s):
        self.model.add_xy(tuple(m), tuple(s))
        self.t += 1
        if not self.bootstrapped_s and self.t > 1:
            if not list(s) == list(self.model.imodel.fmodel.dataset.get_y(self.t - 2)):
                self.bootstrapped_s = True
                
    def update_batch(self, m_list, s_list):
        self.model.add_xy_batch(m_list, s_list)
        self.t += len(m_list)
        self.bootstrapped_s = True
        
    def size(self):
        return self.t
    
    

class ContextNonParametric(NonParametric):
    def __init__(self, conf, context_mode=None, sigma_explo_ratio=0.1, fwd='LWLR', inv='L-BFGS-B', **learner_kwargs):

        NonParametric.__init__(self, conf, sigma_explo_ratio, fwd, inv, **learner_kwargs)
        
        self.context_mode = context_mode
        self.dist_max = np.linalg.norm(self.bounds[0,:] - self.bounds[1,:])
        self.context_dataset = BufferedDataset(self.context_mode['context_n_dims'], 1)
        self.good_context_dataset = BufferedDataset(self.context_mode['context_n_dims'], 1)
        self.eps_dist = 0.05
        
    
    def is_context_new(self, context):
        return self.context_dataset.nn_x(context)[0][0] > self.eps_dist
    
    def did_object_moved(self, s): return abs(s[-1]) > 0.0001
    
    def add_explo_noise(self, x):
        if self.mode == 'explore':
            r = x
            r[self.sigma_expl > 0] = np.random.normal(r[self.sigma_expl > 0], self.sigma_expl[self.sigma_expl > 0])
            res = bounds_min_max(r, self.m_mins, self.m_maxs)
            return res
        else:
            return x
        
    def random_motor_command(self):
        return rand_bounds(np.array([self.m_mins, 
                                     self.m_maxs]))[0]
                                                 
    def competence_point(self, m, s):
        obj_moved = self.did_object_moved(s)
        if obj_moved:
            return competence_dist(s[-self.context_mode["context_n_dims"]:], -s[:self.context_mode["context_n_dims"]], dist_max=self.dist_max)
        else:
            # Find min dist during trajectory between obj and tool (or hand)
            return competence_dist(s[-self.context_mode["context_n_dims"]:], -s[:self.context_mode["context_n_dims"]], dist_max=self.dist_max) - s[2] / self.dist_max
     
    def interest_pt(self, s):
        """
        Interest of this strategy for goal s
        
        """
        context = s[:self.context_mode["context_n_dims"]]
        if len(self.context_dataset) == 0:
            return 1.
        dists, idxs = self.context_dataset.nn_x(context, radius=0.001, k=50)
        idxs = [idxs[k] for k in range(len(idxs)) if dists[k] < 0.001]
        if len(idxs) > 1:
            idxs = sorted(idxs)
            v = np.array([self.context_dataset.get_y(idx)[0] for idx in idxs])
#             print
#             print "idxs", idxs
#             print "comp", v
            n = len(v)
            comp_beg = np.mean(v[:int(float(n)/2.)])
            comp_end = np.mean(v[int(float(n)/2.):])
            return np.abs(comp_end - comp_beg)
        else:
            if len(self.good_context_dataset) > 0:
                i = self.nn_good_context(context)
                #print "nn good context", self.context_dataset.get_x(i)
                return max(1. + competence_dist(context, self.context_dataset.get_x(i)), 0) 
            else:
                return 1.
        
    def nn_good_context(self, context):
        _, idxs = self.good_context_dataset.nn_x(context)
        return self.good_context_dataset.get_y(idxs[0])[0]
                             
    def min_cost_context(self, context):
        # Find min cost point among points with +- same context
        dists, idxs = self.context_dataset.nn_x(context, radius=self.eps_dist, k=100)
        #print dists, idxs, len(self.context_dataset)
        competences = []
        for k in range(len(idxs)):
            if dists[k] >  self.eps_dist:
                break
            s = self.model.imodel.fmodel.dataset.get_y(idxs[k])
            c = self.context_dataset.get_y(idxs[k])[0] + competence_dist(context, s[:self.context_mode["context_n_dims"]], dist_max=self.dist_max)        
            competences.append(c)
        #print "nb of similar contexts", len(competences)
        k = np.argmax(competences)
        return idxs[k], max(competences)
                
    def infer(self, in_dims, out_dims, x):
        if self.t < max(self.model.imodel.fmodel.k, self.model.imodel.k):
            raise ExplautoBootstrapError
        
        if in_dims == self.m_dims and out_dims == self.s_dims:  # forward
            return array(self.model.predict_effect(tuple(x)))
        
        elif in_dims == self.s_dims and out_dims == self.m_dims:  # inverse
            if not self.bootstrapped_s:
                # If only one distinct point has been observed in the sensory space, then we output a random motor command
                return self.random_motor_command()
            else:
                context = x[:self.context_mode["context_n_dims"]]
                #print "nn_context dist", nn_context_dist
                if self.is_context_new(context):
                    #print "context is new"
                    if len(self.good_context_dataset) > 0:
                        i = self.nn_good_context(context)
                        return self.model.imodel.fmodel.dataset.get_x(i)
                    else:
                        return self.random_motor_command()
                else:
                    #print "context is known"
                    i,_ = self.min_cost_context(context)
                    return self.add_explo_noise(self.model.imodel.fmodel.dataset.get_x(i))   
    
    def competence_for_context(self, context):
        if self.is_context_new(context):
            if len(self.good_context_dataset) > 0:
                idx = self.good_context_dataset.nn_x(context)[1][0]
                return self.context_dataset.get_y(self.good_context_dataset.get_y(idx)[0])[0] + competence_dist(context, self.good_context_dataset.get_x(idx), dist_max=self.dist_max)  
            else:
                return -1.
        else:
            #print "context is known"
            _,c = self.min_cost_context(context)
            return c


    def update(self, m, s):
        NonParametric.update(self, m, s)
        self.update_context_dataset(m, s)
        
        
    def update_context_dataset(self, m, s):
        c = self.competence_point(m, s)
        self.context_dataset.add_xy(s[:self.context_mode['context_n_dims']], [c])
        if self.did_object_moved(s): 
            #print "object moved!"
            self.good_context_dataset.add_xy(s[:self.context_mode['context_n_dims']], [len(self.context_dataset)-1])



sensorimotor_models = {
    'nearest_neighbor': (NonParametric, {'default': {'fwd': 'NN', 'inv': 'NN', 'sigma_explo_ratio':0.1},
                                         'exact': {'fwd': 'NN', 'inv': 'NN', 'sigma_explo_ratio':0.}}),
    'WNN': (NonParametric, {'default': {'fwd': 'WNN', 'inv': 'WNN', 'k':20, 'sigma':0.1}}),
    'LWLR-BFGS': (NonParametric, {'default': {'fwd': 'LWLR', 'k':10, 'sigma':0.1, 'inv': 'L-BFGS-B', 'maxfun':50}}),
    'LWLR-CMAES': (NonParametric, {'default': {'fwd': 'LWLR', 'k':10, 'sigma':0.1, 'inv': 'CMAES', 'cmaes_sigma':0.05, 'maxfevals':20}}),
}
