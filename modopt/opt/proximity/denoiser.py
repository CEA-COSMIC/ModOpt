"""Denoiser Operators.

Denoising operators that can be used as a drop-in replacement of proximal
operators in iterative algorithms, making them a "plug-and-play" method.

TODO Add support for VBM4D (video denoising, may be usefull for dynamic data).
TODO Add support for Local Low-Rank Denoising.
"""

from modopt.opt.proximity.base import ProximityParent

BM3D_AVAILABLE = True
try:
    import bm3d
except ImportError:
    BM3D_AVAILABLE = False


class BM3Denoise(ProximityParent):
    """
    BM3D Denoiser.

    This a wrapper around the BM3D Denoiser, using the implementation of
    :cite:`makinen2020`.

    Parameters
    ----------
    noise_std: float, default=1.0
        The noise standart deviation level that will be denoise.
    """

    def __init__(self, noise_std=1.0):
        self.noise_std = noise_std
        self.op = self._op_method
        self.cost = lambda x_val: 0

    def _op_method(self, input_data, extra_factor=1.0):
        """Operator.

        This method returns the input data denoise by the bm3d algorithm.

        Parameters
        ----------
        input_data : numpy.ndarray
            Input data array
        extra_factor : float
            Additional multiplication factor (default is ``1.0``)

        Returns
        -------
        numpy.ndarray
            Denoised data
        """
        return bm3d.bm3d(
            input_data,
            sigma_psd=extra_factor * self.noise_std,
            sigma_arg=bm3d.BM3DStages.ALL_STAGES,
        )
