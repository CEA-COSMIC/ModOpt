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

from modopt.signal.local_pca import local_svd_thresh

class BM3Denoise(ProximityParent):
    """
    BM3D Denoiser.

    This a wrapper around the BM3D Denoiser, using the implementation of
    :cite:`makinen2020`.
    """

    def __init__(self, noise_std=1.0):
        self.noise_std = noise_std
        self.op = self._op_method
        self.cost = lambda x_val: 0

    def _op_method(self, input_data, extra_factor=1.0):
        return bm3d.bm3d(
            input_data,
            sigma_psd=extra_factor * self.noise_std,
            sigma_arg=bm3d.BM3DStages.ALL_STAGES,
        )


class LocalLowRankDenoiser(ProximityParent):
    """Local Low Rank Denoiser.

    This a python implementation of
    "NOise Reduction with DIstribution Corrected" (NORDIC).

    It follows the principles described in :cite:`moeller2021` and
    the MATLAB implementation of  [1]


    Parameters
    ----------
    patch_size: tuple
        the patch size to consider for the local low rank regularisation
    patch_overlap:
        the amount of pixel/voxel overlap of each patch.
        If > 0 the patch are averaged.

    References
    ----------
    .. [1] https://github.com/SteenMoeller/NORDIC_Raw

    """

    def __init__(
        self,
        patch_shape=11,
        patch_overlap=0,
        noise_weights=None,
        threshold_estimation='MPPCA',
        mask=None,
    ):
        self.patch_shape = patch_shape
        self.patch_overlap = patch_overlap
        if threshold_estimation not in {'MPPCA', 'NORDIC'}:
            raise ValueError(
                'Threshold estimation method should be "MPPCA" or "NORDIC"',
            )

        self.threshold_estimation = threshold_estimation
        self.noise_weight = noise_weights
        self.mask = mask

    def _op_method(self, input_data, extra_factor=None):
        # make tuple from patch paratemers
        for attr in {'patch_shape', 'patch_overlap'}:
            if hasattr(getattr(self.attr), '__len__'):
                if len(getattr(self.attr)) != input_data.ndim - 1:
                    raise ValueError(
                        f'{attr} should have one less dim than the input data.',
                    )

        else:
            setattr(self, attr, getattr(self, attr,) * input_data.ndim)

        out, _ = local_svd_thresh(
            input_data,
            self.patch_shape,
            self.patch_overlap,
            mask=self.mask,
            threshold=self.threshold_estimation,
            extra_factor=extra_factor,
        )

        return out
