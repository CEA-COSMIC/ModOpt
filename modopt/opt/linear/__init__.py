"""LINEAR OPERATORS.

This module contains linear operator classes.

:Author: Samuel Farrens <samuel.farrens@cea.fr>
:Author: Pierre-Antoine Comby <pierre-antoine.comby@cea.fr>
"""

from .base import LinearParent, Identity, MatrixOperator, LinearCombo

from .wavelet import WaveletConvolve, WaveletTransform


__all__ = [
    "LinearParent",
    "Identity",
    "MatrixOperator",
    "LinearCombo",
    "WaveletConvolve",
    "WaveletTransform",
]
