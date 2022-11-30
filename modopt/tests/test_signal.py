"""UNIT TESTS FOR SIGNAL.

This module contains unit tests for the modopt.signal module.

:Author: Samuel Farrens <samuel.farrens@cea.fr>
:Author: Pierre-Antoine Comby <pierre-antoine.comby@crans.org>
"""

import numpy as np
import numpy.testing as npt
import pytest

from test_helpers import failparam

from modopt.signal import filter, noise, positivity, svd, validation, wavelet


@pytest.mark.parametrize(
    ("norm", "result"), [(True, 0.24197072451914337), (False, 0.60653065971263342)]
)
def test_gaussian_filter(norm, result):
    """Test gaussian filter."""
    npt.assert_almost_equal(filter.gaussian_filter(1, 1, norm=norm), result)


def test_mex_hat():
    """Test mex_hat."""
    npt.assert_almost_equal(
        filter.mex_hat(2, 1),
        -0.35213905225713371,
    )


def test_mex_hat_dir():
    """Test mex_hat_dir."""
    npt.assert_almost_equal(
        filter.mex_hat_dir(1, 2, 1),
        0.17606952612856686,
    )


class TestNoise:
    """Test noise module."""

    data1 = np.arange(9).reshape(3, 3).astype(float)
    data2 = np.array(
        [[0, 2.0, 2.0], [4.0, 5.0, 10], [11.0, 15.0, 18.0]],
    )
    data3 = np.array(
        [
            [1.62434536, 0.38824359, 1.47182825],
            [1.92703138, 4.86540763, 2.6984613],
            [7.74481176, 6.2387931, 8.3190391],
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
            failparam(data1, "fail", 1, data1),
        ],
    )
    def test_add_noise(self, data, noise_type, sigma, data_noise):
        """Test add_noise."""
        np.random.seed(1)
        npt.assert_almost_equal(
            noise.add_noise(data, sigma=sigma, noise_type=noise_type), data_noise
        )

    @pytest.mark.parametrize(
        ("threshold_type", "result"),
        [("hard", data4), ("soft", data5), failparam("fail", None, raises=ValueError)],
    )
    def test_thresh(self, threshold_type, result):
        """Test threshold."""
        npt.assert_array_equal(noise.thresh(self.data1, 5, threshold_type=threshold_type), result)

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (-1.0, -float(0)),
            (-1, 0),
            (data1 - 5, data5),
            (
                np.array([np.arange(3) - 1, np.arange(2) - 1], dtype=object),
                np.array([np.array([0, 0, 1]),np.array([0, 0])], dtype=object),
            ),
            failparam("-1", None, raises=TypeError),
        ],
    )
    def test_positive(self, value, expected):
        """Test positive."""
        if isinstance(value, np.ndarray) and value.dtype == 'O':
            for v, e in zip(positivity.positive(value), expected):
                npt.assert_array_equal(v, e)
        else:
            npt.assert_array_equal(positivity.positive(value), expected)
