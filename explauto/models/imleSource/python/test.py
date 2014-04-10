import imle
from numpy.random import randn

d = 3
D = 2

model = imle.Imle(in_ndims=d, out_ndims=D, sigma0=0.1, Psi0=[0.01]*2)

# look at imleSource/python/imle.py for all possible parameters and their default values

for _ in range(100):
    model.update(randn(d), randn(D))

# get the number of models
model.number_of_experts

# get the mean of the 1st model
print 'mean of the 1st model' 
print model.get_joint_mu(0)

print 'covariance of the 3nd model'
print model.get_joint_mu(2)

print 'forward prediction'
print model.predict([0., 0.2, -0.1])

print 'inverse prediction'
print model.predict_inverse([0.1, 0.3])
