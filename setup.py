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
    url='https://github.com/cosmostat/ModOpt',
    download_url='https://github.com/cosmostat/ModOpt',
    packages=find_packages(),
    license='MIT',
    description='Modular Optimisation tools for soliving inverse problems.',
    long_description=release_info["__about__"],
    setup_requires=['pytest-runner', ],
    tests_require=['pytest', 'pytest-cov', ],
)
