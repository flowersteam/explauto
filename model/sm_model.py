
import imle as imle_


class Imle(object):
    def __init__(self, **kwargs):
        self.imle=imle_.Imle(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)
    def infer(self, in_dims,out_dims,x):
        return self.gmm.inference(in_dims, out_dims, x).sample().T
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

