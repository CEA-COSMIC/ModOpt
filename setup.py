#! /usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import os

release_info = {}
infopath = os.path.abspath(os.path.join(os.path.dirname(__file__),
                           "modopt", "info.py"))
with open(infopath) as open_file:
    exec(open_file.read(), release_info)

setup(
    name='modopt',
    author='sfarrens',
    author_email='samuel.farrens@cea.fr',
    version=release_info["__version__"],
    url='https://github.com/cea-cosmic/ModOpt',
    download_url='https://github.com/cea-cosmic/ModOpt',
    packages=find_packages(),
<<<<<<< a70614f2f47c0e3dddd8f4b55b1bac64cc668586
    install_requires=['numpy>=1.16.4', 'scipy==1.3.0', 'progressbar2>=3.34.3'],
=======
    install_requires=['numpy>=1.14.1', 'future>=0.16.0', 'scipy>=1.0.0',
                      'progressbar2>=3.34.3', 'scikit-learn>=0.20.3'],
>>>>>>> Adding sklear requirement
    license='MIT',
    description='Modular Optimisation tools for soliving inverse problems.',
    long_description=release_info["__about__"],
    setup_requires=['pytest-runner', ],
    tests_require=['pytest>=5.0.1', 'pytest-cov>=2.7.1', 'pytest-pep8'],
)
