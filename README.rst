ModOpt
======

|travis| |coveralls| |license| |python27| |python35| |python36|

.. |travis| image:: https://travis-ci.org/CEA-COSMIC/ModOpt.svg?branch=master
  :target: https://travis-ci.org/CEA-COSMIC/ModOpt

.. |coveralls| image:: https://coveralls.io/repos/github/CEA-COSMIC/ModOpt/badge.svg
  :target: https://coveralls.io/github/CEA-COSMIC/ModOpt

.. |license| image:: https://img.shields.io/github/license/mashape/apistatus.svg
  :target: https://github.com/CEA-COSMIC/ModOpt/blob/master/LICENCE.txt

.. |python27| image:: https://img.shields.io/badge/python-2.7-green.svg
  :target: https://www.python.org/

.. |python35| image:: https://img.shields.io/badge/python-3.5-green.svg
  :target: https://www.python.org/

.. |python36| image:: https://img.shields.io/badge/python-3.6-green.svg
  :target: https://www.python.org/

:Author: Samuel Farrens `(samuel.farrens@cea.fr) <samuel.farrens@cea.fr>`_

:Version: 1.3.0

:Date: 27/03/2019

:Documentation: |link-to-docs|

.. |link-to-docs| raw:: html

  <a href="https://cea-cosmic.github.io/ModOpt/"
  target="_blank">https://cea-cosmic.github.io/ModOpt/</a>

ModOpt is a series of Modular Optimisation tools for solving inverse problems.

Contents
========

1. `Dependencies`_

   1. `Required Packages`_
   2. `Optional Packages`_

2. `Installation`_

Dependencies
============

Required Packages
-----------------

In order to run the code in this repository the following packages must be
installed:

* |link-to-python| [Last tested with v2.7.15 and v3.7.0]

* |link-to-numpy| [Tested with v1.15.4]

* |link-to-scipy| [Tested with v1.1.0]

* |link-to-future| [Tested with v0.17.1]

* |link-to-progressbar| [Tested with v3.38.0]

.. |link-to-python| raw:: html

  <a href="https://www.python.org/"
  target="_blank">Python</a>

.. |link-to-numpy| raw:: html

  <a href="http://www.numpy.org/"
  target="_blank">Numpy</a>

.. |link-to-scipy| raw:: html

  <a href="http://www.scipy.org/"
  target="_blank">Scipy</a>

.. |link-to-future| raw:: html

  <a href="http://python-future.org/quickstart.html"
  target="_blank">Future</a>

.. |link-to-progressbar| raw:: html

  <a href="https://progressbar-2.readthedocs.io/en/latest/"
  target="_blank">Progressbar 2</a>

Optional Packages
-----------------

The following packages can optionally be installed to add extra functionality:

* |link-to-astropy| [Last tested with v3.0.5]

* |link-to-matplotlib| [Last tested with v3.0.2]

* |link-to-skimage| [Last tested with v0.14.1]

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

.. |link-to-termcolor| raw:: html

  <a href="https://pypi.python.org/pypi/termcolor"
  target="_blank">Termcolor</a>

Installation
============

To clone the ModOpt repository from GitHub run the following command:

.. code-block:: bash

  $ git clone https://github.com/cea-cosmic/ModOpt

To install using `easy_install` run the following command:

.. code-block:: bash

  $ easy_install modopt

To install using `pip` run the following command:

.. code-block:: bash

  $ pip install modopt
