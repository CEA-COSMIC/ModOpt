"""UNIT TESTS FOR SIGNAL.

This module contains unit tests for the modopt.signal module.

:Authors:
    Samuel Farrens <samuel.farrens@cea.fr>
    Pierre-Antoine Comby <pierre-antoine.comby@cea.fr>
"""

import numpy as np
import numpy.testing as npt
import pytest
from test_helpers import failparam

from modopt.signal import filter, noise, positivity, svd, validation, wavelet


class TestFilter:
    """Test filter module."""

    @pytest.mark.parametrize(
        ("norm", "result"), [(True, 0.24197072451914337), (False, 0.60653065971263342)]
    )
    def test_gaussian_filter(self, norm, result):
        """Test gaussian filter."""
        npt.assert_almost_equal(filter.gaussian_filter(1, 1, norm=norm), result)

    def test_mex_hat(self):
        """Test mexican hat filter."""
        npt.assert_almost_equal(
            filter.mex_hat(2, 1),
            -0.35213905225713371,
        )

    def test_mex_hat_dir(self):
        """Test directional mexican hat filter."""
        npt.assert_almost_equal(
            filter.mex_hat_dir(1, 2, 1),
            0.17606952612856686,
        )


class TestNoise:
    """Test noise module."""

    data1 = np.arange(9).reshape(3, 3).astype(float)
    data2 = np.array(
        [[0, 3.0, 4.0], [6.0, 9.0, 8.0], [14.0, 14.0, 17.0]],
    )
    data3 = np.array(
        [
            [0.3455842, 1.8216181, 2.3304371],
            [1.6968428, 4.9053559, 5.4463746],
            [5.4630468, 7.5811181, 8.3645724],
        ]
    )
    data4 = np.array([[0, 0, 0], [0, 0, 5.0], [6.0, 7.0, 8.0]])
    data5 = np.array(
        [[0, 0, 0], [0, 0, 0], [1.0, 2.0, 3.0]],
    )

    @pytest.mark.parametrize(
        ("data", "noise_type", "sigma", "data_noise"),
        [
            (data1, "poisson", 1, data2),
            (data1, "gauss", 1, data3),
            (data1, "gauss", (1, 1, 1), data3),
            failparam(data1, "fail", 1, data1, raises=ValueError),
        ],
    )
    def test_add_noise(self, data, noise_type, sigma, data_noise):
        """Test add_noise."""
        rng = np.random.default_rng(1)
        npt.assert_almost_equal(
            noise.add_noise(data, sigma=sigma, noise_type=noise_type, rng=rng),
            data_noise,
        )

    @pytest.mark.parametrize(
        ("threshold_type", "result"),
        [("hard", data4), ("soft", data5), failparam("fail", None, raises=ValueError)],
    )
    def test_thresh(self, threshold_type, result):
        """Test threshold."""
        npt.assert_array_equal(
            noise.thresh(self.data1, 5, threshold_type=threshold_type), result
        )


class TestPositivity:
    """Test positivity module."""

    data1 = np.arange(9).reshape(3, 3).astype(float)
    data4 = np.array([[0, 0, 0], [0, 0, 5.0], [6.0, 7.0, 8.0]])
    data5 = np.array(
        [[0, 0, 0], [0, 0, 0], [1.0, 2.0, 3.0]],
    )

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (-1.0, -float(0)),
            (-1, 0),
            (data1 - 5, data5),
            (
                np.array([np.arange(3) - 1, np.arange(2) - 1], dtype=object),
                np.array([np.array([0, 0, 1]), np.array([0, 0])], dtype=object),
            ),
            failparam("-1", None, raises=TypeError),
        ],
    )
    def test_positive(self, value, expected):
        """Test positive."""
        if isinstance(value, np.ndarray) and value.dtype == "O":
            for v, e in zip(positivity.positive(value), expected):
                npt.assert_array_equal(v, e)
        else:
            npt.assert_array_equal(positivity.positive(value), expected)


class TestSVD:
    """Test for svd module."""

    @pytest.fixture
    def data(self):
        """Initialize test data."""
        data1 = np.arange(18).reshape(9, 2).astype(float)
        data2 = np.arange(32).reshape(16, 2).astype(float)
        data3 = np.array(
            [
                np.array(
                    [
                        [-0.01744594, -0.61438865],
                        [-0.08435304, -0.50397984],
                        [-0.15126014, -0.39357102],
                        [-0.21816724, -0.28316221],
                        [-0.28507434, -0.17275339],
                        [-0.35198144, -0.06234457],
                        [-0.41888854, 0.04806424],
                        [-0.48579564, 0.15847306],
                        [-0.55270274, 0.26888188],
                    ]
                ),
                np.array([42.23492742, 1.10041151]),
                np.array(
                    [
                        [-0.67608034, -0.73682791],
                        [0.73682791, -0.67608034],
                    ]
                ),
            ],
            dtype=object,
        )
        data4 = np.array(
            [
                [-1.05426832e-16, 1.0],
                [2.0, 3.0],
                [4.0, 5.0],
                [6.0, 7.0],
                [8.0, 9.0],
                [1.0e1, 1.1e1],
                [1.2e1, 1.3e1],
                [1.4e1, 1.5e1],
                [1.6e1, 1.7e1],
            ]
        )

        data5 = np.array(
            [
                [0.49815487, 0.54291537],
                [2.40863386, 2.62505584],
                [4.31911286, 4.70719631],
                [6.22959185, 6.78933678],
                [8.14007085, 8.87147725],
                [10.05054985, 10.95361772],
                [11.96102884, 13.03575819],
                [13.87150784, 15.11789866],
                [15.78198684, 17.20003913],
            ]
        )
        return (data1, data2, data3, data4, data5)

    @pytest.fixture
    def svd0(self, data):
        """Compute SVD of first data sample."""
        return svd.calculate_svd(data[0])

    def test_find_n_pc(self, data):
        """Test find number of principal component."""
        npt.assert_equal(
            svd.find_n_pc(svd.svd(data[1])[0]),
            2,
            err_msg="Incorrect number of principal components.",
        )

    def test_n_pc_fail_non_square(self):
        """Test find_n_pc."""
        npt.assert_raises(ValueError, svd.find_n_pc, np.arange(3))

    def test_calculate_svd(self, data, svd0):
        """Test calculate_svd."""
        errors = []
        for i, name in enumerate("USV"):
            try:
                npt.assert_almost_equal(svd0[i], data[2][i])
            except AssertionError:
                errors.append(name)
        if errors:
            raise AssertionError("Incorrect SVD calculation for: " + ", ".join(errors))

    @pytest.mark.parametrize(
        ("n_pc", "idx_res"),
        [(None, 3), (1, 4), ("all", 0), failparam("fail", 1, raises=ValueError)],
    )
    def test_svd_thresh(self, data, n_pc, idx_res):
        """Test svd_tresh."""
        npt.assert_almost_equal(
            svd.svd_thresh(data[0], n_pc=n_pc),
            data[idx_res],
        )

    def test_svd_tresh_invalid_type(self):
        """Test svd_tresh failure."""
        npt.assert_raises(TypeError, svd.svd_thresh, 1)

    @pytest.mark.parametrize("operator", [lambda x: x, failparam(0, raises=TypeError)])
    def test_svd_thresh_coef(self, data, operator):
        """Test svd_tresh_coef."""
        npt.assert_almost_equal(
            svd.svd_thresh_coef(data[0], operator, 0),
            data[0],
            err_msg="Incorrect SVD coefficient tresholding",
        )

    # TODO test_svd_thresh_coef_fast


class TestValidation:
    """Test validation Module."""

    array33 = np.arange(9).reshape(3, 3)

    def test_transpose_test(self):
        """Test transpose_test."""
        npt.assert_equal(
            validation.transpose_test(
                lambda x_val, y_val: x_val.dot(y_val),
                lambda x_val, y_val: x_val.dot(y_val.T),
                self.array33.shape,
                x_args=self.array33,
                rng=2,
            ),
            None,
        )


class TestWavelet:
    """Test Wavelet Module."""

    @pytest.fixture
    def data(self):
        """Set test parameter values."""
        data1 = np.arange(9).reshape(3, 3).astype(float)
        data2 = np.arange(36).reshape(4, 3, 3).astype(float)
        data3 = np.array(
            [
                [
                    [6.0, 20, 26.0],
                    [36.0, 84.0, 84.0],
                    [90, 164.0, 134.0],
                ],
                [
                    [78.0, 155.0, 134.0],
                    [225.0, 408.0, 327.0],
                    [270, 461.0, 350],
                ],
                [
                    [150, 290, 242.0],
                    [414.0, 732.0, 570],
                    [450, 758.0, 566.0],
                ],
                [
                    [222.0, 425.0, 350],
                    [603.0, 1056.0, 813.0],
                    [630, 1055.0, 782.0],
                ],
            ]
        )

        data4 = np.array(
            [
                [6496.0, 9796.0, 6544.0],
                [9924.0, 14910, 9924.0],
                [6544.0, 9796.0, 6496.0],
            ]
        )

        data5 = np.array(
            [
                [[0, 1.0, 4.0], [3.0, 10, 13.0], [6.0, 19.0, 22.0]],
                [[3.0, 10, 13.0], [24.0, 46.0, 40], [45.0, 82.0, 67.0]],
                [[6.0, 19.0, 22.0], [45.0, 82.0, 67.0], [84.0, 145.0, 112.0]],
            ]
        )
        return (data1, data2, data3, data4, data5)

    @pytest.mark.parametrize(
        ("idx_data", "idx_filter", "idx_res", "filter_rot"),
        [(0, 1, 2, False), (1, 1, 3, True)],
    )
    def test_filter_convolve(self, data, idx_data, idx_filter, idx_res, filter_rot):
        """Test filter_convolve."""
        npt.assert_almost_equal(
            wavelet.filter_convolve(
                data[idx_data], data[idx_filter], filter_rot=filter_rot
            ),
            data[idx_res],
            err_msg="Inccorect filter comvolution.",
        )

    def test_filter_convolve_stack(self, data):
        """Test filter_convolve_stack."""
        npt.assert_almost_equal(
            wavelet.filter_convolve_stack(data[0], data[0]),
            data[4],
            err_msg="Inccorect filter stack comvolution.",
        )
