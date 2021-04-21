# ModOpt

<img width=400 src="docs/source/modopt_logo.png">

| Usage | Development | Release |
| ----- | ----------- | ------- |
| [![docs](https://img.shields.io/badge/docs-Sphinx-blue)](https://cea-cosmic.github.io/ModOpt/) | [![build](https://github.com/CEA-COSMIC/modopt/workflows/CI/badge.svg)](https://github.com/CEA-COSMIC/modopt/actions?query=workflow%3ACI) | [![release](https://img.shields.io/github/v/release/CEA-COSMIC/modopt)](https://github.com/CEA-COSMIC/modopt/releases/latest) |
| [![license](https://img.shields.io/github/license/CEA-COSMIC/modopt)](https://github.com/CEA-COSMIC/modopt/blob/master/LICENCE.txt) | [![deploy](https://github.com/CEA-COSMIC/modopt/workflows/CD/badge.svg)](https://github.com/CEA-COSMIC/modopt/actions?query=workflow%3ACD) | [![pypi](https://img.shields.io/pypi/v/modopt)](https://pypi.org/project/modopt/) |
| [![wemake-python-styleguide](https://img.shields.io/badge/style-wemake-000000.svg)](https://github.com/wemake-services/wemake-python-styleguide) | [![codecov](https://codecov.io/gh/CEA-COSMIC/modopt/branch/master/graph/badge.svg?token=XHJIQXV7AX)](https://codecov.io/gh/CEA-COSMIC/modopt) | [![python](https://img.shields.io/pypi/pyversions/modopt)](https://www.python.org/downloads/source/) |
| [![contribute](https://img.shields.io/badge/contribute-read-lightgrey)](https://github.com/CEA-COSMIC/modopt/blob/master/CONTRIBUTING.md) | [![CodeFactor](https://www.codefactor.io/repository/github/CEA-COSMIC/modopt/badge)](https://www.codefactor.io/repository/github/CEA-COSMIC/modopt) | |
| [![coc](https://img.shields.io/badge/conduct-read-lightgrey)](https://github.com/CEA-COSMIC/modopt/blob/master/CODE_OF_CONDUCT.md) | [![Updates](https://pyup.io/repos/github/CEA-COSMIC/modopt/shield.svg)](https://pyup.io/repos/github/CEA-COSMIC/ModOpt/) | |

ModOpt is a series of **Modular Optimisation** tools for solving inverse problems.

See [documentation](https://CEA-COSMIC.github.io/ModOpt/) for more details.

## Installation

To install using `pip` run the following command:

```bash
  $ pip install modopt
```

To clone the ModOpt repository from GitHub run the following command:

```bash
  $ git clone https://github.com/CEA-COSMIC/ModOpt.git
```

## Dependencies

All packages required by ModOpt should be installed automatically. Optional packages, however, will need to be installed manually.

### Required Packages

In order to run the code in this repository the following packages must be
installed:

* [Python](https://www.python.org/) [> 3.6]
* [importlib_metadata](https://importlib-metadata.readthedocs.io/en/latest/) [==3.7.0]
* [Numpy](http://www.numpy.org/) [==1.19.5]
* [Scipy](http://www.scipy.org/) [==1.5.4]
* [Progressbar 2](https://progressbar-2.readthedocs.io/) [==3.53.1]

### Optional Packages

The following packages can optionally be installed to add extra functionality:

* [Astropy](http://www.astropy.org/)
* [Matplotlib](http://matplotlib.org/)
* [Scikit-Image](https://scikit-image.org/)
* [Scikit-Learn](https://scikit-learn.org/)
* [Termcolor](https://pypi.python.org/pypi/termcolor)

For (partial) GPU compliance the following packages can also be installed.
Note that none of these are required for running on a CPU.

* [CuPy](https://cupy.dev/)
* [Torch](https://pytorch.org/)
* [TensorFlow](https://www.tensorflow.org/)

## Citation

If you use ModOpt in a scientific publication, we would appreciate citations to the following paper:

[PySAP: Python Sparse Data Analysis Package for multidisciplinary image processing](https://www.sciencedirect.com/science/article/pii/S2213133720300561), S. Farrens et al., Astronomy and Computing 32, 2020

The BibTeX citation is the following:
```
@Article{farrens2020pysap,
  title={{PySAP: Python Sparse Data Analysis Package for multidisciplinary image processing}},
  author={Farrens, S and Grigis, A and El Gueddari, L and Ramzi, Z and Chaithya, GR and Starck, S and Sarthou, B and Cherkaoui, H and Ciuciu, P and Starck, J-L},
  journal={Astronomy and Computing},
  volume={32},
  pages={100402},
  year={2020},
  publisher={Elsevier}
}
```
