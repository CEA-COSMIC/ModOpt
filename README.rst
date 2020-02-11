ModOpt
======

|travis| |coveralls| |license| |python35| |python36| |python37| |python38| |pypi|

.. |travis| image:: https://travis-ci.org/CEA-COSMIC/ModOpt.svg?branch=master
  :target: https://travis-ci.org/CEA-COSMIC/ModOpt

.. |coveralls| image:: https://coveralls.io/repos/github/CEA-COSMIC/ModOpt/badge.svg
  :target: https://coveralls.io/github/CEA-COSMIC/ModOpt

.. |license| image:: https://img.shields.io/github/license/mashape/apistatus.svg
  :target: https://github.com/CEA-COSMIC/ModOpt/blob/master/LICENCE.txt

.. |python35| image:: https://img.shields.io/badge/python-3.5-green.svg
  :target: https://www.python.org/

.. |python36| image:: https://img.shields.io/badge/python-3.6-green.svg
  :target: https://www.python.org/

.. |python37| image:: https://img.shields.io/badge/python-3.7-green.svg
  :target: https://www.python.org/

.. |python38| image:: https://img.shields.io/badge/python-3.8-green.svg
  :target: https://www.python.org/

.. |pypi| image:: https://badge.fury.io/py/modopt.svg
  :target: https://badge.fury.io/py/modopt


:Author: Samuel Farrens `(samuel.farrens@cea.fr) <samuel.farrens@cea.fr>`_

:Version: 1.4.2

:Release Date: 14/02/2020

:Documentation: |link-to-docs|

:Repository: |link-to-repo|

.. |link-to-docs| raw:: html

  <a href="https://cea-cosmic.github.io/ModOpt/"
  target="_blank">https://cea-cosmic.github.io/ModOpt/</a>

.. |link-to-repo| raw:: html

  <a href="https://github.com/CEA-COSMIC/ModOpt"
  target="_blank">https://github.com/CEA-COSMIC/ModOpt</a>

ModOpt is a series of Modular Optimisation tools for solving inverse problems.

Contents
========

1. `Dependencies`_
2. `Installation`_

Dependencies
============

Required Packages
-----------------

In order to run the code in this repository the following packages must be
installed:

* |link-to-python| [Last tested with v3.7.0]

* |link-to-numpy| [Tested with v1.16.4]

* |link-to-scipy| [Tested with v1.3.0]

* |link-to-progressbar| [Tested with v3.42.0]

.. |link-to-python| raw:: html

  <a href="https://www.python.org/"
  target="_blank">Python</a>

.. |link-to-numpy| raw:: html

  <a href="http://www.numpy.org/"
  target="_blank">Numpy</a>

.. |link-to-scipy| raw:: html

  <a href="http://www.scipy.org/"
  target="_blank">Scipy</a>

.. |link-to-progressbar| raw:: html

  <a href="https://progressbar-2.readthedocs.io/en/latest/"
  target="_blank">Progressbar 2</a>

Optional Packages
-----------------

The following packages can optionally be installed to add extra functionality:

* |link-to-astropy| [Last tested with v3.2.1]

* |link-to-matplotlib| [Last tested with v3.1.1]

* |link-to-skimage| [Requires >=v0.16.0]

* |link-to-sklearn| [Requires >=v0.21.3]

* |link-to-termcolor| [Last tested with v1.1.0]

.. |link-to-astropy| raw:: html

  <a href="http://www.astropy.org/"
  target="_blank">Astropy</a>

.. |link-to-matplotlib| raw:: html

  <a href="http://matplotlib.org/"
  target="_blank">Matplotlib</a>

.. |link-to-skimage| raw:: html

  <a href="https://scikit-image.org/"
  target="_blank">Scikit-Image</a>

.. |link-to-sklearn| raw:: html

  <a href="https://scikit-learn.org/"
  target="_blank">Scikit-Learn</a>

.. |link-to-termcolor| raw:: html

  <a href="https://pypi.python.org/pypi/termcolor"
  target="_blank">Termcolor</a>

Installation
============

To install using `pip` run the following command:

.. code-block:: bash

  $ pip install modopt

To install using `easy_install` run the following command:

.. code-block:: bash

  $ easy_install modopt

To clone the ModOpt repository from GitHub run the following command:

.. code-block:: bash

  $ git clone https://github.com/cea-cosmic/ModOpt
