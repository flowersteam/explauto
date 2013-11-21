
import imle as imle_


class Imle(object):
    def __init__(self, **kwargs):
        self.imle=imle_.Imle(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)
    def infer(self, in_dims,out_dims,x):
        if in_dims == self.s_dims and out_dims==self.m_dims:
            try:
                return self.imle.predict_inverse(x.flatten())[0].reshape(-1,1)
            except RuntimeError as e:
                print e
                return self.imle.to_gmm().inference(in_dims, out_dims, x).sample().T
        else:
            return self.imle.to_gmm().inference(in_dims, out_dims, x).sample().T
    def update(self, m, s):
        self.imle.update(m.flatten(),s.flatten())

class Imle_Gmm(Imle):
    def __init__(self, **kwargs):
        Imle.__init__(self, **kwargs)
        self.update_gmm()
    def update_gmm(self):
        self.gmm=self.imle.to_gmm()
    def infer(self, in_dims,out_dims,x):
        return self.gmm.inference(in_dims, out_dims, x).sample().T
    def update(self, m, s):
        super(Imle_Gmm,self).update(m,s)
        self.update_gmm()

