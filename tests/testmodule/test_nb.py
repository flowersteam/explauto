import pytest
import os

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

path = os.path.abspath(__file__)
libdir_name = os.path.dirname(os.path.dirname(os.path.dirname(path)))


nbfilename_list = [
	#os.path.join(libdir_name,'notebook/full_tutorial.ipynb'),
	os.path.join(libdir_name,'notebook/05 Customising configurations, environments and agents.ipynb'),
	]

@pytest.mark.parametrize("notebook_filename", nbfilename_list)
def test_notebook(notebook_filename):
	with open(notebook_filename) as f:
		nb = nbformat.read(f, as_version=4)
	ep = ExecutePreprocessor()
	ep.preprocess(nb, {'metadata': {'path': './'}})
