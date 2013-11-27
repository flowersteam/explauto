.. models documentation master file, created by
   sphinx-quickstart on Wed Mar 21 14:11:24 2012.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::
   :maxdepth: 2
   
   reference.rst

Purpose
=======

Python package such as scipy and scikit-learn implements troves of regression, optimization and machine learning algorithms. Yet, some are lacking and some implementations are not flexible enough for our needs. In this small library, we implement machine learning algorithms we absolutely could not find anywhere else in a convenient form.

Currently, models implements a data structure for nearest neighbors search, LLWR (Locally Linear Weighted Regression), and optimization data structures to inverse forward models.

30 Seconds Tutorial
===================

In order to learn the mapping between two multidimensional feature spaces, just type: 

import models


Core concepts
=============

There are three kind of core objects in the library : *datasets*, *models*.

Dataset
-------

Dataset holds the data and provides nearest neighbors methods on the data. They can be used on 

.. literalinclude:: ../../examples/docs/dataset.py

You can also use the class WeightedDataset, that allows you to set customs weights on the input and output dimensions, which will impact the nearest neighbors and regression routines.

.. literalinclude:: ../../examples/docs/wdataset.py
   
Models
------

Models implement regression methods. 

Forward models implement the following interface :
 
.. literalinclude:: ../../models/forward/forward.py

An example usage is :

.. literalinclude:: ../../examples/docs/forward0.py

Forward model can also be instanciated from existing datasets :

.. literalinclude:: ../../examples/docs/forward1.py

Optimizers 
----------

To optimize the model parameters to the current data, general-purpose techniques are provided, such as cross-validation.