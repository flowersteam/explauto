# Explauto: A library to study, model and simulated autonomous exploration in virtual and robotics agents #

Explauto is a framework developed in the [inria FLOWERS](https://flowers.inria.fr/) team to provide a common interface for the implementation of active sensorimotor learning algorithm.

Explauto allows the standardized but flexible implementation of:

* Sensorimotor active learning models (cognitive level)
* Agent motor and sensory primitives (agent level)
* Virtual and robotics setups (environment level)

All interfaces are written in Python to allow for fast development, easy deployment and quick scripting by non-necessary expert developers. Python is highly interoperable with other programming language such as C++, Matlab, etc ... and a number of tutorial explaing how to bind the library with third-party softwares using othe programming languages are provided.

It is crossed-platform and has been tested on Linux, Windows and Mac OS.

Do not hesitate to contact us if you want to get involved!

## Scientific grounding ##
Explauto's scientific roots trace back from Intelligent Adaptive Curiosity algorithmic architecture (REF Kaplan Oudeyer (lien vers hal)), which has been extended to a more general family of autonomous exploration architecture by (REF Baranes) and recently expressed as a compact and unified formalism (REF Moulin-Frier Oudeyer). 


## Documentation ##

The full Explauto documentation on a html format can be found [here](https://bitbucket.org/ClementMF/Explauto/). It provides tutorials, examples and a complete API.

## Installation ##

Before you start building PyPot, you need to make sure that the following packages are already installed on your computer:

* [python](http://www.python.org) 2.7
* [numpy](http://www.numpy.org)

Once it is done, you can build and install PyPot with the classical:

    cd Explauto
    python setup.py install

For more details on the installation procedure, please refer to the [installation section](https://bitbucket.org/ClementMF/Explauto/intro.html#installation) of the documentation.
