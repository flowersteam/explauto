import os
from distutils.core import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "toolbox",
    version = "1.0",
    author = "Fabien Benureau",
    author_email = "fabien.benureau@gmail.com",
    description = ("Shared routines across projects"),
    license = "Not Yet Decided.",
    keywords = "python toolbox",
    packages=['toolbox'],
    #long_description=read('README'),
    classifiers=[],
)