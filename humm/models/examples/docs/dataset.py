from models.dataset import Dataset

# Creating a dataset of dimension 2 in input and 3 in output
dset = Dataset(2, 3)

# Adding datapoints
dset.add_xy([0.0, 1.0], [ 1.0, 2.0, 0.0])
dset.add_xy([1.0, 0.0], [ 0.0, 0.0, 2.0])
dset.add_xy([2.0,-1.0], [-1.0,-2.0, 4.0])

# Nearest neighbors queries on input, requesting 2 neighbors
dset.nn_x([0.2, 0.5], 2)
# Nearest neighbors queries on output, requesting 1 neighbors
dist, index = dset.nn_y([1.0, 1.0, 1.0], 1)

# Retrieving the nearest output of [1.0, 1.0, 1.0]
print dset.get_y(index[0])
# Retrieving the nearest datapoint
print dset.get_xy(index[0])