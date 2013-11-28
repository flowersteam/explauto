import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "models",
    version = "0.3",
    author = "Fabien Benureau and Clement Moulin-Frier",
    author_email = "fabien.benureau@inria.fr, clement.moulin-frier@gmail.com",
    description = ("A python implementation of LWR and other classic forward and inverse models"),
    license = "Not Yet Decided.",
    keywords = "lwr machine-learning",
    url = "flowers.inria.fr",
    packages=['models', 'models.samples', 
                        'models.forward', 
                        'models.inverse', 
                        'models.testbed', 
                        'models.plots'],
    #long_description=read('README'),
    classifiers=[],
)