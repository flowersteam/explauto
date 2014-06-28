#!/usr/bin/env python

import re
import sys

from setuptools import setup, find_packages


def version():
    with open('explauto/_version.py') as f:
        return re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read()).group(1)

extra = {}
if sys.version_info >= (3,):
    extra['use_2to3'] = True

setup(name='explauto',
      version=version(),
      packages=find_packages(),

      install_requires=['numpy', 'scipy', 'scikit-learn', 'pandas'],

      extra_require={
          'diva': ['pymatlab'],
          'pypot': ['pypot'],
          'imle': [],
          'doc': ['sphinx', 'sphinx-bootstrap-theme'],
      },

      setup_requires=['setuptools_git >= 0.3', ],

      #   include_package_data=True,
      #   exclude_package_data={'': ['README', '.gitignore']},

      zip_safe=True,

      author='Clement Moulin-Frier, Pierre Rouanet',
      author_email='clement.moulinfrier@gmail.com',
      description='Python Library for Autonomous Exploration',
      url='https://github.com/flowersteam/explauto',
      license='GNU GENERAL PUBLIC LICENSE Version 3',
      **extra
      )
