from models.dataset import WeightedDataset

# Creating a weighted dataset of dimension 2 in input and 3 in output,
# with custom weights
dset = WeightedDataset(2, 3, weights_x = [0.1, 1.0],
                             weights_y = [0.0, 1.0, 10.0])

# Adding datapoints
dset.add_xy([0.0, 1.0], [ 10.0, 2.0, 0.0])
dset.add_xy([1.0, 0.0], [  0.0, 0.0, 2.0])

# This will print [ 0. 1.]
dist, index = dset.nn_x([0.0, 0.56], 1)
print dset.get_x(index[0])

# This will print [ 0. 0. 2.]
dist, index = dset.nn_y([10.0, 2.0, 2.0], 1)
print dset.get_y(index[0])
