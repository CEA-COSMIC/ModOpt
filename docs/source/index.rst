.. modopt documentation master file, created by
   sphinx-quickstart on Mon Oct 24 16:46:22 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ModOpt Documentation
======================

:Author: Samuel Farrens <samuel.farrens@cea.fr>

:Version: 1.1.5

:Date: 11/04/2018

ModOpt is a series of Modular Optimisation tools for solving inverse problems.

Contents
========

1. `Dependencies`_

   1. `Required Packages`_
   2. `Optional Packages`_

2. `Installation`_

3. `Package Contents`_

Dependencies
============

Required Packages
-----------------

In order to run the code in this repository the following packages must be
installed:

* |link-to-python| [Tested with 2.7.11 and 3.6.3]

* |link-to-numpy| [Tested with v1.13.3]

* |link-to-scipy| [Tested with v1.0.0]

* |link-to-future| [Tested with v0.16.0]

* |link-to-astropy| [Tested with v2.0.2]

* |link-to-progressbar| [Tested with v3.34.3]


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

.. |link-to-astropy| raw:: html

  <a href="http://www.astropy.org/"
  target="_blank">Astropy</a>

.. |link-to-progressbar| raw:: html

  <a href="https://progressbar-2.readthedocs.io/en/latest/"
  target="_blank">Progressbar 2</a>

Optional Packages
-----------------

The following packages can optionally be installed to add extra functionality:

* |link-to-matplotlib| [Tested with v2.1.0]

* |link-to-termcolor| [Tested with v1.1.0]

.. |link-to-matplotlib| raw:: html

  <a href="http://matplotlib.org/"
  target="_blank">Matplotlib</a>

.. |link-to-termcolor| raw:: html

  <a href="https://pypi.python.org/pypi/termcolor"
  target="_blank">Termcolor</a>

Installation
============

To clone the modopt repository from GitHub run the following command:

.. code-block:: bash

  $ git clone https://github.com/cea-cosmic/modopt

To install using `easy_install` run the following command:

.. code-block:: bash

  $ easy_install modopt

To install using `pip` run the following command:

.. code-block:: bash

  $ pip install modopt

Package Contents
================

.. toctree::
   :numbered:
   :maxdepth: 3

   examples
   modopt
