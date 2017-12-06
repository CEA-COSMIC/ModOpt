from setuptools import setup, find_packages
from modopt import __version__

setup(
    name='modopt',
    author='sfarrens',
    author_email='samuel.farrens@cea.fr',
    version=__version__,
    url='https://github.com/cosmostat/ModOpt',
    download_url='https://github.com/cosmostat/ModOpt',
    packages=find_packages(),
    license='MIT',
    description='Modular Optimisation tools for soliving inverse problems.',
    long_description=open('README.txt').read(),
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
)
