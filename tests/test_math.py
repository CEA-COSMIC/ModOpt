"""UNIT TESTS FOR MATH.

This module contains unit tests for the modopt.math module.

:Authors:
    Samuel Farrens <samuel.farrens@cea.fr>
    Pierre-Antoine Comby <pierre-antoine.comby@cea.fr>
"""

import pytest
from test_helpers import failparam, skipparam

import numpy as np
import numpy.testing as npt


from modopt.math import convolve, matrix, metrics, stats

try:
    import astropy
except ImportError:  # pragma: no cover
    ASTROPY_AVAILABLE = False
else:  # pragma: no cover
    ASTROPY_AVAILABLE = True
try:
    from skimage.metrics import structural_similarity as compare_ssim
except ImportError:  # pragma: no cover
    SKIMAGE_AVAILABLE = False
else:
    SKIMAGE_AVAILABLE = True

rng = np.random.default_rng(1)


class TestConvolve:
    """Test convolve functions."""

    array233 = np.arange(18).reshape((2, 3, 3))
    array233_1 = array233 + 1
    result_astropy = np.array(
        [
            [210.0, 201.0, 210.0],
            [129.0, 120.0, 129.0],
            [210.0, 201.0, 210.0],
        ]
    )
    result_scipy = np.array(
        [
            [
                [14.0, 35.0, 38.0],
                [57.0, 120.0, 111.0],
                [110.0, 197.0, 158.0],
            ],
            [
                [518.0, 845.0, 614.0],
                [975.0, 1578.0, 1137.0],
                [830.0, 1331.0, 950.0],
            ],
        ]
    )

    result_rot_kernel = np.array(
        [
            [
                [66.0, 115.0, 82.0],
                [153.0, 240.0, 159.0],
                [90.0, 133.0, 82.0],
            ],
            [
                [714.0, 1087.0, 730.0],
                [1125.0, 1698.0, 1131.0],
                [738.0, 1105.0, 730.0],
            ],
        ]
    )

    @pytest.mark.parametrize(
        ("input_data", "kernel", "method", "result"),
        [
            skipparam(
                array233[0],
                array233_1[0],
                "astropy",
                result_astropy,
                cond=not ASTROPY_AVAILABLE,
                reason="astropy not available",
            ),
            failparam(
                array233[0], array233_1, "astropy", result_astropy, raises=ValueError
            ),
            failparam(
                array233[0], array233_1[0], "fail!", result_astropy, raises=ValueError
            ),
            (array233[0], array233_1[0], "scipy", result_scipy[0]),
        ],
    )
    def test_convolve(self, input_data, kernel, method, result):
        """Test convolve function."""
        npt.assert_allclose(convolve.convolve(input_data, kernel, method), result)

    @pytest.mark.parametrize(
        ("result", "rot_kernel"),
        [
            (result_scipy, False),
            (result_rot_kernel, True),
        ],
    )
    def test_convolve_stack(self, result, rot_kernel):
        """Test convolve stack function."""
        npt.assert_allclose(
            convolve.convolve_stack(
                self.array233, self.array233_1, rot_kernel=rot_kernel
            ),
            result,
        )


class TestMatrix:
    """Test matrix module."""

    array3 = np.arange(3)
    array33 = np.arange(9).reshape((3, 3))
    array23 = np.arange(6).reshape((2, 3))
    gram_schmidt_out = (
        np.array(
            [
                [0, 1.0, 2.0],
                [3.0, 1.2, -6e-1],
                [-1.77635684e-15, 0, 0],
            ]
        ),
        np.array(
            [
                [0, 0.4472136, 0.89442719],
                [0.91287093, 0.36514837, -0.18257419],
                [-1.0, 0, 0],
            ]
        ),
    )

    @pytest.fixture(scope="module")
    def pm_instance(self, request):
        """Power Method instance."""
        pm = matrix.PowerMethod(
            lambda x_val: x_val.dot(x_val.T),
            self.array33.shape,
            verbose=True,
            rng=np.random.default_rng(0),
        )
        return pm

    @pytest.mark.parametrize(
        ("return_opt", "output"),
        [
            ("orthonormal", gram_schmidt_out[1]),
            ("orthogonal", gram_schmidt_out[0]),
            ("both", gram_schmidt_out),
            failparam("fail!", gram_schmidt_out, raises=ValueError),
        ],
    )
    def test_gram_schmidt(self, return_opt, output):
        """Test gram schmidt."""
        npt.assert_allclose(
            matrix.gram_schmidt(self.array33, return_opt=return_opt), output
        )

    def test_nuclear_norm(self):
        """Test nuclear norm."""
        npt.assert_almost_equal(
            matrix.nuclear_norm(self.array33),
            15.49193338482967,
        )

    def test_project(self):
        """Test project."""
        npt.assert_array_equal(
            matrix.project(self.array3, self.array3 + 3),
            np.array([0, 2.8, 5.6]),
        )

    def test_rot_matrix(self):
        """Test rot_matrix."""
        npt.assert_allclose(
            matrix.rot_matrix(np.pi / 6),
            np.array([[0.8660254, -0.5], [0.5, 0.8660254]]),
        )

    def test_rotate(self):
        """Test rotate."""
        npt.assert_array_equal(
            matrix.rotate(self.array33, np.pi / 2),
            np.array([[2, 5, 8], [1, 4, 7], [0, 3, 6]]),
        )

        npt.assert_raises(ValueError, matrix.rotate, self.array23, np.pi / 2)

    def test_power_method(self, pm_instance, value=1):
        """Test power method."""
        npt.assert_almost_equal(pm_instance.spec_rad, value)
        npt.assert_almost_equal(pm_instance.inv_spec_rad, 1 / value)


class TestMetrics:
    """Test metrics module."""

    data1 = np.arange(49).reshape(7, 7)
    mask = np.ones(data1.shape)
    ssim_res = 0.8958315888566867
    ssim_mask_res = 0.8023827544418249
    snr_res = 10.134554256920536
    psnr_res = 14.860761791850397
    mse_res = 0.03265305507330247
    nrmse_res = 0.31136678840022625

    @pytest.mark.skipif(not SKIMAGE_AVAILABLE, reason="skimage not installed")
    @pytest.mark.parametrize(
        ("data1", "data2", "result", "mask"),
        [
            (data1, data1**2, ssim_res, None),
            (data1, data1**2, ssim_mask_res, mask),
            failparam(data1, data1, None, 1, raises=ValueError),
        ],
    )
    def test_ssim(self, data1, data2, result, mask):
        """Test ssim."""
        npt.assert_almost_equal(metrics.ssim(data1, data2, mask=mask), result)

    @pytest.mark.skipif(SKIMAGE_AVAILABLE, reason="skimage installed")
    def test_ssim_fail(self):
        """Test ssim."""
        npt.assert_raises(ImportError, metrics.ssim, self.data1, self.data1)

    @pytest.mark.parametrize(
        ("metric", "data", "result", "mask"),
        [
            (metrics.snr, data1, snr_res, None),
            (metrics.snr, data1, snr_res, mask),
            (metrics.psnr, data1, psnr_res, None),
            (metrics.psnr, data1, psnr_res, mask),
            (metrics.mse, data1, mse_res, None),
            (metrics.mse, data1, mse_res, mask),
            (metrics.nrmse, data1, nrmse_res, None),
            (metrics.nrmse, data1, nrmse_res, mask),
            failparam(metrics.snr, data1, snr_res, "maskfail", raises=ValueError),
        ],
    )
    def test_metric(self, metric, data, result, mask):
        """Test snr."""
        npt.assert_almost_equal(metric(data, data**2, mask=mask), result)


class TestStats:
    """Test stats module."""

    array33 = np.arange(9).reshape(3, 3)
    array233 = np.arange(18).reshape(2, 3, 3)

    @pytest.mark.skipif(not ASTROPY_AVAILABLE, reason="astropy not installed")
    @pytest.mark.parametrize(
        ("norm", "result"),
        [
            (
                "max",
                np.array(
                    [
                        [0.36787944, 0.60653066, 0.36787944],
                        [0.60653066, 1.0, 0.60653066],
                        [0.36787944, 0.60653066, 0.36787944],
                    ]
                ),
            ),
            (
                "sum",
                np.array(
                    [
                        [0.07511361, 0.1238414, 0.07511361],
                        [0.1238414, 0.20417996, 0.1238414],
                        [0.07511361, 0.1238414, 0.07511361],
                    ]
                ),
            ),
            failparam("fail", None, raises=ValueError),
        ],
    )
    def test_gaussian_kernel(self, norm, result):
        """Test Gaussian kernel."""
        npt.assert_allclose(
            stats.gaussian_kernel(self.array33.shape, 1, norm=norm), result
        )

    @pytest.mark.skipif(ASTROPY_AVAILABLE, reason="astropy installed")
    def test_import_astropy(self):
        """Test missing astropy."""
        npt.assert_raises(ImportError, stats.gaussian_kernel, self.array33.shape, 1)

    def test_mad(self):
        """Test mad."""
        npt.assert_equal(stats.mad(self.array33), 2.0)

    def test_sigma_mad(self):
        """Test sigma_mad."""
        npt.assert_almost_equal(
            stats.sigma_mad(self.array33),
            2.9651999999999998,
        )

    @pytest.mark.parametrize(
        ("data1", "data2", "method", "result"),
        [
            (array33, array33 + 2, "starck", 12.041199826559248),
            failparam(array33, array33, "fail", 0, raises=ValueError),
            (array33, array33 + 2, "wiki", 42.110203695399477),
        ],
    )
    def test_psnr(self, data1, data2, method, result):
        """Test PSNR."""
        npt.assert_almost_equal(stats.psnr(data1, data2, method=method), result)

    def test_psnr_stack(self):
        """Test psnr stack."""
        npt.assert_almost_equal(
            stats.psnr_stack(self.array233, self.array233 + 2),
            12.041199826559248,
        )

        npt.assert_raises(ValueError, stats.psnr_stack, self.array33, self.array33)
