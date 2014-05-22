.. _installation:

Installation
============

Via Python Packages
-------------------

The Explauto core is entirely written in Python. So, the install process should be rather straightforward. You can directly install it via easy_install or pip::

    pip install explauto

or::

    easy_install explauto

The up to date archive can also be directly downloaded `here <https://pypi.python.org/pypi/explauto/>`_.

Make sure to also install all the dependecies required by the extra environment or models (e.g. pypot environment or imle models) you want to use. See the section :ref:`extra` for details.

From the source code
--------------------

You can also install it from the source. You can clone/fork our repo directly on `github <https://github.com/flowersteam/explauto/>`_.

Before you start building Explauto, you need to make sure that the following packages are already installed on your computer:

* `python <http://www.python.org>`_ 2.7 or 3
* `numpy <http://www.numpy.org>`_
* `scipy <http://www.scipy.org>`_
* `scikit-learn <http://scikit-learn.org>`_
* `pandas <http://pandas.pydata.org>`_,


Other optional packages may be installed depending on extra environment, interest, and sensorimotor models you need:

* `pypot <https://github.com/poppy-project/pypot>`_ for using dynamixel based robot as enviroment
* `diva <http://www.bu.edu/speechlab/software/diva-source-code/>`_ (for the diva....)

If you want to build the documentation from the source:

* `sphinx <http://sphinx-doc.org/index.html>`_

Once it is done, you can build and install explauto with the classical::

    cd explauto
    python setup.py build
    python setup.py install

Testing your install
--------------------

You can test if the installation went well with::

    python -m "import explauto"

If you have any trouble during the installation process, please refer to the :doc:`FAQ </FAQ>`. If this does not solve your problem, please report the issue directly on the issue tracker of the repository.
