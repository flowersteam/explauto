import numpy as np

from ..utils import rand_bounds
from .interest_model import InterestModel
from .competences import competence_exp, competence_dist
from ..models.dataset import BufferedDataset as Dataset


class RandomInterest(InterestModel):
    def __init__(self, conf, expl_dims):
        InterestModel.__init__(self, expl_dims)

        self.bounds = conf.bounds[:, expl_dims]
        self.ndims = self.bounds.shape[1]

    def sample(self):
        return rand_bounds(self.bounds).flatten()

    def update(self, xy, ms):
        pass

    def sample_given_context(self, c, c_dims):
        '''
        Sample randomly on dimensions not in context
            c: context value on c_dims dimensions, not used
            c_dims: w.r.t sensori space dimensions
        '''
        return self.sample()[list(set(range(self.ndims)) - set(c_dims))]
    

class MiscRandomInterest(RandomInterest):
    """
    Add some features to the RandomInterest random babbling class.
    
    Allows to query the recent interest in the whole space,
    the recent competence on the babbled points in the whole space, 
    the competence around a given point based on a mean of the knns.   
    
    """
    def __init__(self, 
                 conf, 
                 expl_dims,
                 competence_measure,
                 win_size,
                 competence_mode,
                 k,
                 progress_mode,
                 mode="sg"):
        
        RandomInterest.__init__(self, conf, expl_dims)
        
        self.competence_measure = competence_measure
        self.win_size = win_size
        self.competence_mode = competence_mode
        self.dist_max = np.linalg.norm(self.bounds[0,:] - self.bounds[1,:])
        self.k = k
        self.progress_mode = progress_mode
        self.mode = mode
        self.data_xc = Dataset(len(expl_dims), 1)
        self.data_sr = Dataset(len(expl_dims), 0)
        self.current_progress = 0.
        self.current_interest = 0.
              
            
    def add_xc(self, x, c):
        self.data_xc.add_xy(x, [c])
        
    def add_sr(self, x):
        self.data_sr.add_xy(x)
        
    def update_interest(self, i):
        self.current_progress += (1. / self.win_size) * (i - self.current_progress)
        self.current_interest = abs(self.current_progress)

    def update(self, xy, ms, snnp=None, sp=None):
        if self.mode == "sg":
            c = self.competence_measure(xy[self.expl_dims], ms[self.expl_dims], dist_max=self.dist_max)
            if self.progress_mode == 'local':
                interest = self.interest_xc(xy[self.expl_dims], c)
                self.update_interest(interest)
                #print 's', ms[self.expl_dims]
            elif self.progress_mode == 'global':
                pass
            
            self.add_xc(xy[self.expl_dims], c)
            self.add_sr(ms[self.expl_dims])
        elif self.mode == "sg_snn":
            sgnnp = xy[self.expl_dims] + snnp
            c = self.competence_measure(sgnnp, ms[self.expl_dims], dist_max=self.dist_max)
            #print "competence", c, sgnnp, ms[self.expl_dims]
            
            interest = self.interest_xc(xy[self.expl_dims], c)
            self.update_interest(interest)
            self.add_xc(xy[self.expl_dims], c)
            
            snnpr = ms[self.expl_dims] - snnp
            self.add_sr(snnpr)
        elif self.mode == "sp":
            c = self.competence_measure(sp, ms[self.expl_dims], dist_max=self.dist_max)
            #print "competence", c, sgnnp, ms[self.expl_dims]
            
            interest = self.interest_xc(xy[self.expl_dims], c)
            self.update_interest(interest)
            self.add_xc(xy[self.expl_dims], c)
            self.add_sr(sp)
            
        return interest
    
    def n_points(self):
        return len(self.data_xc)
    
    def competence_global(self, mode='sw'):
        if self.n_points() > 0:
            if mode == 'all':
                return np.mean(self.data_c)
            elif mode == 'sw':
                idxs = range(self.n_points())[- self.win_size:]
                return np.mean([self.data_xc.get_y(idx) for idx in idxs])
            else:
                raise NotImplementedError
        else:
            return 0.
        
    def mean_competence_pt(self, x):
        if self.n_points() > self.k: 
            _, idxs = self.data_xc.nn_x(x, k=self.k)
            return np.mean([self.data_xc.get_y(idx) for idx in idxs])
        else:
            return self.competence()
                
    def interest_xc(self, x, c):
        """
        Interest of point x with competence c 
        
        """
        if self.n_points() > 0:
            if self.mode == "sg" or self.mode == "sg_snn":
                idx_sg_NN = self.data_xc.nn_x(x, k=1)[1][0]
                sr_NN = self.data_sr.get_x(idx_sg_NN)
                c_old = competence_dist(x, sr_NN, dist_max=self.dist_max)
                    
    #             print 
    #             print "x", x
    #             print "sr_NN", sr_NN
#                 print "c_old", c_old 
#                 print "c_new", c
#                 print "interest", c - c_old
    
                #return c - c_old
                return np.abs(c - c_old)
            elif self.mode == "sp":
                idx_sg_NN = self.data_xc.nn_x(x, k=1)[1][0]
                c_old = self.data_xc.get_y(idx_sg_NN)[0]
#                 print "c_old", c_old 
#                 print "c_new", c
                return c - c_old
                
        else:
            return 0.
#         mean_local_comp = self.mean_competence_pt(x)
#         if mean_local_comp == 0:
#             return np.abs(c - mean_local_comp)
#         else:
#             return np.abs((c - mean_local_comp)/mean_local_comp)
        
    def interest_pt(self, x):
        if self.n_points() > self.k:
            _, idxs = self.data_xc.nn_x(x, k=self.k)
            idxs = sorted(idxs)
            v = [self.data_xc.get_y(idx) for idx in idxs]
            n = len(v)
            comp_beg = np.mean(v[:int(float(n)/2.)])
            comp_end = np.mean(v[int(float(n)/2.):])
            return np.abs(comp_end - comp_beg)
        else:
            return self.interest_global()
            
    def interest_global(self): 
        if self.n_points() < 2:
            return 0.
        else:
            idxs = range(self.n_points())[- self.win_size:]
            v = [self.data_xc.get_y(idx) for idx in idxs]
            n = len(v)
            comp_beg = np.mean(v[:int(float(n)/2.)])
            comp_end = np.mean(v[int(float(n)/2.):])
            return np.abs(comp_end - comp_beg)
        
    def competence(self):
        return self.competence_global()
        
    def interest(self):
        if self.progress_mode == 'local':
            return self.current_interest
        elif self.progress_mode == 'global':
            return self.interest_global()
        else:
            raise NotImplementedError
        
        
        

class ContextRandomInterest(MiscRandomInterest):
    """
    Add some features to the RandomInterest random babbling class.
    
    Allows to query the recent interest in the whole space,
    the recent competence on the babbled points in the whole space, 
    the competence around a given point based on a mean of the knns.   
    
    """
    def __init__(self, 
                 conf, 
                 expl_dims,
                 win_size,
                 competence_mode,
                 k,
                 progress_mode,
                 mode="sg",
                 context_mode=None):
        
        self.context_mode = context_mode
        
        MiscRandomInterest.__init__(self,
                                     conf, 
                                     expl_dims,
                                     self.competence_measure,
                                     win_size,
                                     competence_mode,
                                     k,
                                     progress_mode,
                                     mode=mode)
        
        self.eps_dist = 0.05
        
              
    def competence_measure(self, msg, ms, dist_max):
        m = ms[:-len(self.expl_dims)]
        context = ms[self.expl_dims][:self.context_mode["context_n_dims"]]
        #print "context", context
        s = ms[self.expl_dims][-self.context_mode["context_n_dims"]:]
        #print "s", s
        obj_moved = abs(s[-1]) > 0.0001
        if obj_moved:
            return competence_dist(s, -context, dist_max=dist_max)
        else:
            # Find min dist during trajectory between obj and tool (or hand)
#             print "context dist", competence_dist(s, -context, dist_max=dist_max)
#             print "hand or tool dists during mov", [competence_dist([m[ix], m[iy]], context, dist_max=dist_max) for (ix, iy) in [(0,3), (1,4), (2,5)]]
            return competence_dist(s, -context, dist_max=dist_max) - ms[self.expl_dims][2] / dist_max
        
    def interest_xc(self, x, c):
        """
        Interest of point x with competence c 
        
        """
        if self.n_points() > 0:
            if self.mode == "sg" or self.mode == "sg_snn":
                idx_sg_NN = self.data_xc.nn_x(x, k=1)[1][0]
                dists, idxs = self.data_xc.nn_x(self.data_xc.get_x(idx_sg_NN), radius=0.001, k=10)
                c_old = min([self.data_xc.get_y(idxs[k])[0] for k in range(len(idxs)) if dists[k] < 0.001])
    #             print 
    #             print "x", x
    #             print "sr_NN", sr_NN
#                 print "c_old", c_old 
#                 print "c_new", c
#                 print "interest", c - c_old
    
                #return c - c_old
                return np.abs(c - c_old)
        else:
            return 0.
                
    def novelty_bonus(self):
        return 1.
    
    def update(self, xy, ms, snnp=None, sp=None):
        if self.mode == "sg":
            c = self.competence_measure(xy, ms, dist_max=self.dist_max)
            #print "competence:", c
            if self.progress_mode == 'local':
                #print xy, self.expl_dims
                interest = self.interest_xc(xy[self.expl_dims], c)
                #print "interest", interest
                self.update_interest(interest)
                #print 's', ms[self.expl_dims]
            elif self.progress_mode == 'global':
                pass
            
            self.add_xc(xy[self.expl_dims], c)
            self.add_sr(ms[self.expl_dims])
            
        elif self.mode == "sp":
            c = self.competence_measure(sp, ms[self.expl_dims], dist_max=self.dist_max)
            #print "competence", c, sgnnp, ms[self.expl_dims]
            
            interest = self.interest_xc(xy[self.expl_dims], c)
            self.update_interest(interest)
            self.add_xc(xy[self.expl_dims], c)
            self.add_sr(sp)
            
        return interest
    
    
        
interest_models = {'random': (RandomInterest, {'default': {}}),
                   'misc_random': (MiscRandomInterest, {'default': 
                       {'competence_measure': competence_dist,
                                   'win_size': 100,
                                   'competence_mode': 'knn',
                                   'k': 100,
                                   'progress_mode': 'local'}})}
