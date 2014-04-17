from cmodels import *
a = cDataset(2,2)
#b = cLWLRForwardModel(2,2, dset = a)
print a.size
print a
a.add_xy(np.array(range(2)), np.array(range(2, 4)))
print 'bla3'