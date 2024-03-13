"""
Test for base module.

:Authors:
    Samuel Farrens <samuel.farrens@cea.fr>
    Pierre-Antoine Comby <pierre-antoine.comby@cea.fr>
"""

import numpy as np
import numpy.testing as npt
import pytest
from test_helpers import failparam, skipparam

from modopt.base import backend, np_adjust, transform, types
from modopt.base.backend import LIBRARIES


class TestNpAdjust:
    """Test for npadjust."""

    array33 = np.arange(9).reshape((3, 3))
    array233 = np.arange(18).reshape((2, 3, 3))
    arraypad = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 1, 2, 0],
            [0, 3, 4, 5, 0],
            [0, 6, 7, 8, 0],
            [0, 0, 0, 0, 0],
        ]
    )

    def test_rotate(self):
        """Test rotate."""
        npt.assert_array_equal(
            np_adjust.rotate(self.array33),
            np.rot90(np.rot90(self.array33)),
            err_msg="Incorrect rotation.",
        )

    def test_rotate_stack(self):
        """Test rotate_stack."""
        npt.assert_array_equal(
            np_adjust.rotate_stack(self.array233),
            np.rot90(self.array233, k=2, axes=(1, 2)),
            err_msg="Incorrect stack rotation.",
        )

    @pytest.mark.parametrize(
        "padding",
        [
            1,
            [1, 1],
            np.array([1, 1]),
            failparam("1", raises=ValueError),
        ],
    )
    def test_pad2d(self, padding):
        """Test pad2d."""
        npt.assert_equal(np_adjust.pad2d(self.array33, padding), self.arraypad)

    def test_fancy_transpose(self):
        """Test fancy transpose."""
        npt.assert_array_equal(
            np_adjust.fancy_transpose(self.array233),
            np.array(
                [
                    [[0, 3, 6], [9, 12, 15]],
                    [[1, 4, 7], [10, 13, 16]],
                    [[2, 5, 8], [11, 14, 17]],
                ]
            ),
            err_msg="Incorrect fancy transpose",
        )

    def test_ftr(self):
        """Test ftr."""
        npt.assert_array_equal(
            np_adjust.ftr(self.array233),
            np.array(
                [
                    [[0, 3, 6], [9, 12, 15]],
                    [[1, 4, 7], [10, 13, 16]],
                    [[2, 5, 8], [11, 14, 17]],
                ]
            ),
            err_msg="Incorrect fancy transpose: ftr",
        )

    def test_ftl(self):
        """Test fancy transpose left."""
        npt.assert_array_equal(
            np_adjust.ftl(self.array233),
            np.array(
                [
                    [[0, 9], [1, 10], [2, 11]],
                    [[3, 12], [4, 13], [5, 14]],
                    [[6, 15], [7, 16], [8, 17]],
                ]
            ),
            err_msg="Incorrect fancy transpose: ftl",
        )


class TestTransforms:
    """Test for the transform module."""

    cube = np.arange(16).reshape((4, 2, 2))
    map = np.array([[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]])
    matrix = np.array([[0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15]])
    layout = (2, 2)
    fail_layout = (3, 3)

    @pytest.mark.parametrize(
        ("func", "indata", "layout", "outdata"),
        [
            (transform.cube2map, cube, layout, map),
            failparam(transform.cube2map, np.eye(2), layout, map, raises=ValueError),
            (transform.map2cube, map, layout, cube),
            (transform.map2matrix, map, layout, matrix),
            (transform.matrix2map, matrix, matrix.shape, map),
        ],
    )
    def test_map(self, func, indata, layout, outdata):
        """Test cube2map."""
        npt.assert_array_equal(
            func(indata, layout),
            outdata,
        )
        if func.__name__ != "map2matrix":
            npt.assert_raises(ValueError, func, indata, self.fail_layout)

    def test_cube2matrix(self):
        """Test cube2matrix."""
        npt.assert_array_equal(
            transform.cube2matrix(self.cube),
            self.matrix,
        )

    def test_matrix2cube(self):
        """Test matrix2cube."""
        npt.assert_array_equal(
            transform.matrix2cube(self.matrix, self.cube[0].shape),
            self.cube,
            err_msg="Incorrect transformation: matrix2cube",
        )


class TestType:
    """Test for type module."""

    data_list = list(range(5))  # noqa: RUF012
    data_int = np.arange(5)
    data_flt = np.arange(5).astype(float)

    @pytest.mark.parametrize(
        ("data", "checked"),
        [
            (1.0, 1.0),
            (1, 1.0),
            (data_list, data_flt),
            (data_int, data_flt),
            failparam("1.0", 1.0, raises=TypeError),
        ],
    )
    def test_check_float(self, data, checked):
        """Test check float."""
        npt.assert_array_equal(types.check_float(data), checked)

    @pytest.mark.parametrize(
        ("data", "checked"),
        [
            (1.0, 1),
            (1, 1),
            (data_list, data_int),
            (data_flt, data_int),
            failparam("1", None, raises=TypeError),
        ],
    )
    def test_check_int(self, data, checked):
        """Test check int."""
        npt.assert_array_equal(types.check_int(data), checked)

    @pytest.mark.parametrize(
        ("data", "dtype"), [(data_flt, np.integer), (data_int, np.floating)]
    )
    def test_check_npndarray(self, data, dtype):
        """Test check_npndarray."""
        npt.assert_raises(
            TypeError,
            types.check_npndarray,
            data,
            dtype=dtype,
        )

    def test_check_callable(self):
        """Test callable."""
        npt.assert_raises(TypeError, types.check_callable, 1)


@pytest.mark.parametrize(
    "backend_name",
    [
        skipparam(name, cond=LIBRARIES[name] is None, reason=f"{name} not installed")
        for name in LIBRARIES
    ],
)
def test_tf_backend(backend_name):
    """Test Modopt computational backends."""
    xp, checked_backend_name = backend.get_backend(backend_name)
    if checked_backend_name != backend_name or xp != LIBRARIES[backend_name]:
        raise AssertionError(f"{backend_name} get_backend fails!")
    xp_input = backend.change_backend(np.array([10, 10]), backend_name)
    if (
        backend.get_array_module(LIBRARIES[backend_name].ones(1))
        != backend.LIBRARIES[backend_name]
        or backend.get_array_module(xp_input) != LIBRARIES[backend_name]
    ):
        raise AssertionError(f"{backend_name} backend fails!")
