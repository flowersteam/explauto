# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import sys
from numpy import *
from matplotlib.pyplot import *

sys.path.append('../../environment/pendule/')
sys.path.append('../../imleSource/build/3_2/lib/')
sys.path.append('../../imleSource/python/')
sys.path.append('../../')
sys.path.append('../../model/')

# <codecell>

import gmminf

import random as rd

import simple_lip as lip


XMIN=-pi
XMAX=pi

VMIN=-2.5
VMAX=2.5

UMIN=-0.25
UMAX=0.25


def generate(XRANGE, DXRANGE, URANGE):

    xt=rd.uniform(XRANGE[0],XRANGE[1])
    dxt=rd.uniform(DXRANGE[0],DXRANGE[1])
    ut=rd.uniform(URANGE[0],URANGE[1])
    res=lip.simulate([xt,dxt],[ut])
    xt1=res[0]
    dxt1=res[1]

    #return Xt,Vt,Ut,Xt+1,Vt+1
    return [xt,dxt,ut,xt1,dxt1]


def build(XRANGE, DXRANGE, URANGE, NB=10):
    res=[(generate(XRANGE,DXRANGE,URANGE)) for i in range(NB) ]

    return array(res)
    

rd.seed()



#Ça devrait être assez explicite...
res= build([XMIN,XMAX],[VMIN,VMAX],[UMIN,UMAX],10000)

# <codecell>

# res.shape

# <codecell>

import imle
model = imle.Imle(in_ndims=3, out_ndims=2, sigma0=(pi/24), Psi0=[0.1]*2)

# <codecell>

[model.update(r[0:3], r[3:]) for r in res]

# <codecell>

print model.number_of_experts

# <codecell>

# clf()
# proj=[3,4]
# #plot(zeros(10), hist[-11:-1,0], 'o')
# COLORS = ['r', 'g', 'b', 'k', 'm']*10000
# a=subplot(111)
# els = model.to_gmm().inference([],proj,[]).get_display_ellipses2D(COLORS)
# for el in els:
#     a.add_patch(el)
# axis('tight')



# <codecell>

# a=subplot(111)
# els = gmminf.get_display_ellipses2D(COLORS)
# for el in els:
#     a.add_patch(el)
# axis('tight')

# # <codecell>

# inds = argsort(gmminf.weights_)

# # <codecell>

# gmminf.means_[inds[-1]]

# <codecell>

# print res.shape[0]
# test_inds = randint(0,res.shape[0],100)
# print res
test_inds = [rd.randint(0,res.shape[0]) for i in range(100)]

# print res[test_inds,0:3]

# print test_inds
# for r in res[test_inds,0:3]:
#     print r,r.reshape(-1,1)

# stop

preds = [model.to_gmm().inference(range(0,3), range(3,5), r.reshape(-1,1)).sample() for r in res[test_inds,0:3]]

# <codecell>

preds = array(preds).reshape(-1,2)

# <codecell>

# print preds

print mean([linalg.norm(preds[i,:] - res[t, -2:]) for i,t in enumerate(test_inds)])


# gmmpend=model.to_gmm().inference(range(0,3), range(3,5),array([0.0,0.0,0.0]))
gmmpend=model.inference(range(0,3), range(3,5),array([0.0,0.0,0.0]))

print "proba"
# print gmmpend.probability(array([0.0,0.0,0.0,0.0,0.1]))

# print gmmpend.probability(array([0.0,0.0]))

# <codecell>

# print preds

# <codecell>


