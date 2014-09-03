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

Make sure to also install all the dependencies required by the extra environment or models (e.g. pypot environment or imle models) you want to use.

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

If you want to build the documentation from the source:

* `sphinx <http://sphinx-doc.org/index.html>`_

Once it is done, you can build and install explauto with the classical::

    cd explauto
    python setup.py build
    python setup.py install

As a developer
--------------

As Explauto is still in development, a good way to keep with an updated version is to use it in `development mode <https://pythonhosted.org/setuptools/setuptools.html#development-mode>`_. To do that, after cloning the `repository <https://github.com/flowersteam/explauto>`_, you can use the following command line::

    cd explauto
    python setup.py develop

This will symlink your cloned repository to your site-packages directory (run the last command as a super user if you are on Linux). When pulling the latest modification from the repo, they will automatically be "installed".

Testing your install
--------------------

You can test if the installation went well with::

    python -c "import explauto"

If you have any trouble during the installation process, please report the issue directly on the `issue tracker <https://github.com/flowersteam/explauto/issues>`_ of the repository.
