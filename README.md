# Explauto: A library to study, model, and simulate autonomous exploration in virtual and robotic agents #

Explauto is a framework developed in the [Inria FLOWERS](https://flowers.inria.fr/) research team which provide a common interface for the implementation of active sensorimotor learning algorithm.

Explauto provides a high-level API for an easy definition of:

* Virtual and robotics setups (Environment level)
* Sensorimotor learning iterative models (Sensorimotor level)
* Active choice of sensorimotor experiments (Interest level)

It is crossed-platform and has been tested on Linux, Windows and Mac OS. Do not hesitate to contact us if you want to get involved!

## Documentation ##

### Scientific grounding ###


Explauto's scientific roots trace back from Intelligent Adaptive Curiosity algorithmic architecture [[Oudeyer 07]](http://hal.inria.fr/hal-00793610/en), which has been extended to a more general family of autonomous exploration architecture by [[Baranes 13]](http://www.pyoudeyer.com/ActiveGoalExploration-RAS-2013.pdf) and recently expressed as a compact and unified formalism [[Moulin-Frier 13]](http://hal.inria.fr/hal-00860641). We strongly recommend to read this [short introduction](http://flowersteam.github.io/explauto/about.html) into developmental robotics before going through the tutorials.

### Tutorials ###

* [Setting a basic experiment](http://nbviewer.ipython.org/github/flowersteam/explauto/blob/master/notebook/01%20Running%20a%20basic%20experiment..ipynb?create=1)
*
*

### API ###

The Explauto documentation on a html format can be found [here](http://flowersteam.github.io/explauto/).


## Installation ##

Explauto is available via pip. It can thus be installed with the classical:

    pip install explauto
    
or:
    
    easy_install explauto

The core of explauto depends of the following packages:

* [python](http://www.python.org) 2.7 or 3.*
* [numpy](http://www.numpy.org)
* [scipy](http://www.scipy.org)
* [scikit-learn](http://scikit-learn.org/)
* [pandas](http://pandas.pydata.org)

For more details, please refer to the [installation section](http://flowersteam.github.io/explauto/installation.html) of the documentation.
