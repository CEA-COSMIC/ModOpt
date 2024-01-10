#!/usr/bin/env python3
"""Wavelet operator, using either scipy filter or pywavelet."""
import warnings

import numpy as np

from modopt.base.types import check_float
from modopt.signal.wavelet import filter_convolve_stack

from .base import LinearParent

pywt_available = True
try:
    import pywt
    from joblib import Parallel, cpu_count, delayed
except ImportError:
    pywt_available = False


class WaveletConvolve(LinearParent):
    """Wavelet Convolution Class.

    This class defines the wavelet transform operators via convolution with
    predefined filters.

    Parameters
    ----------
    filters: numpy.ndarray
        Array of wavelet filter coefficients
    method : str, optional
        Convolution method (default is ``'scipy'``)

    See Also
    --------
    LinearParent : parent class
    modopt.signal.wavelet.filter_convolve_stack : wavelet filter convolution

    """

    def __init__(self, filters, method='scipy'):

        self._filters = check_float(filters)
        self.op = lambda input_data: filter_convolve_stack(
            input_data,
            self._filters,
            method=method,
        )
        self.adj_op = lambda input_data: filter_convolve_stack(
            input_data,
            self._filters,
            filter_rot=True,
            method=method,
        )



class WaveletTransform(LinearParent):
    """
    2D and 3D wavelet transform class.

    This is a light wrapper around PyWavelet, with multicoil support.

    Parameters
    ----------
    wavelet_name: str
        the wavelet name to be used during the decomposition.
    shape: tuple[int,...]
        Shape of the input data. The shape should be a tuple of length 2 or 3.
        It should not contains coils or batch dimension.
    nb_scales: int, default 4
        the number of scales in the decomposition.
    n_batchs: int, default 1
        the number of channel/ batch dimension
    n_jobs: int, default 1
        the number of cores to use for multichannel.
    backend: str, default "threading"
        the backend to use for parallel multichannel linear operation.
    verbose: int, default 0
        the verbosity level.

    Attributes
    ----------
    nb_scale: int
        number of scale decomposed in wavelet space.
    n_jobs: int
        number of jobs for parallel computation
    n_batchs: int
        number of coils use f
    backend: str
        Backend use for parallel computation
    verbose: int
        Verbosity level
    """

    def __init__(
        self,
        wavelet_name,
        shape,
        level=4,
        n_batch=1,
        n_jobs=1,
        decimated=True,
        backend="threading",
        mode="symmetric",
    ):
        if not pywt_available:
            raise ImportError(
                "PyWavelet and/or joblib are not available. Please install it to use WaveletTransform."
            )
        if wavelet_name not in pywt.wavelist(kind="all"):
            raise ValueError(
                "Invalid wavelet name. Availables are  ``pywt.waveletlist(kind='all')``"
            )

        self.wavelet = wavelet_name
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = shape
        self.n_jobs = n_jobs
        self.mode = mode
        self.level = level
        if not decimated:
            raise NotImplementedError(
                "Undecimated Wavelet Transform is not implemented yet."
            )
        ca, *cds = pywt.wavedecn_shapes(
            self.shape, wavelet=self.wavelet, mode=self.mode, level=self.level
        )
        self.coeffs_shape = [ca] + [s for cd in cds for s in cd.values()]

        if len(shape) > 1:
            self.dwt = pywt.wavedecn
            self.idwt = pywt.waverecn
            self._pywt_fun = "wavedecn"
        else:
            self.dwt = pywt.wavedec
            self.idwt = pywt.waverec
            self._pywt_fun = "wavedec"

        self.n_batch = n_batch
        if self.n_batch == 1 and self.n_jobs != 1:
            warnings.warn("Making n_jobs = 1 for WaveletTransform as n_batchs = 1")
            self.n_jobs = 1
        self.backend = backend
        n_proc = self.n_jobs
        if n_proc < 0:
            n_proc = cpu_count() + self.n_jobs + 1

    def op(self, data):
        """Define the wavelet operator.

        This method returns the input data convolved with the wavelet filter.

        Parameters
        ----------
        data: ndarray or Image
            input 2D data array.

        Returns
        -------
        coeffs: ndarray
            the wavelet coefficients.
        """
        if self.n_batch > 1:
            coeffs, self.coeffs_slices, self.raw_coeffs_shape = zip(
                *Parallel(
                    n_jobs=self.n_jobs, backend=self.backend, verbose=self.verbose
                )(delayed(self._op)(data[i]) for i in np.arange(self.n_batch))
            )
            coeffs = np.asarray(coeffs)
        else:
            coeffs, self.coeffs_slices, self.raw_coeffs_shape = self._op(data)
        return coeffs

    def _op(self, data):
        """Single coil wavelet transform."""
        return pywt.ravel_coeffs(
            self.dwt(data, mode=self.mode, level=self.level, wavelet=self.wavelet)
        )

    def adj_op(self, coeffs):
        """Define the wavelet adjoint operator.

        This method returns the reconstructed image.

        Parameters
        ----------
        coeffs: ndarray
            the wavelet coefficients.

        Returns
        -------
        data: ndarray
            the reconstructed data.
        """
        if self.n_batch > 1:
            images = Parallel(
                n_jobs=self.n_jobs, backend=self.backend, verbose=self.verbose
            )(
                delayed(self._adj_op)(coeffs[i], self.coeffs_shape[i])
                for i in np.arange(self.n_batch)
            )
            images = np.asarray(images)
        else:
            images = self._adj_op(coeffs)
        return images

    def _adj_op(self, coeffs):
        """Single coil inverse wavelet transform."""
        return self.idwt(
            pywt.unravel_coeffs(
                coeffs, self.coeffs_slices, self.raw_coeffs_shape, self._pywt_fun
            ),
            wavelet=self.wavelet,
            mode=self.mode,
        )
