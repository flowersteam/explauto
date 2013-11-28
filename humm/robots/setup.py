import os
from distutils.core import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "robots",
    version = "0.1",
    author = "Fabien Benureau",
    author_email = "fabien.benureau@inria.fr",
    description = ("Simple simulated robots with common interface."),
    license = "LGPL OpenScience",
    keywords = "simulation robots",
    url = "flowers.inria.fr",
    packages=['robots'],
    #long_description=read('README'),
    classifiers=[],
)