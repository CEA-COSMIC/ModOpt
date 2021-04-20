# -*- coding: utf-8 -*-

"""UNIT TESTS FOR BASE.

This module contains unit tests for the modopt.base module.

:Author: Samuel Farrens <samuel.farrens@cea.fr>

"""

from builtins import range
from unittest import TestCase, skipIf

import numpy as np
import numpy.testing as npt

from modopt.base import np_adjust, transform, types
from modopt.base.backend import LIBRARIES, change_backend, get_array_module, get_backend


class NPAdjustTestCase(TestCase):
    """Test case for np_adjust module."""

    def setUp(self):
        """Set test parameter values."""
        self.data1 = np.arange(9).reshape((3, 3))
        self.data2 = np.arange(18).reshape((2, 3, 3))
        self.data3 = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 1, 2, 0],
            [0, 3, 4, 5, 0],
            [0, 6, 7, 8, 0],
            [0, 0, 0, 0, 0],
        ])

    def tearDown(self):
        """Unset test parameter values."""
        self.data1 = None
        self.data2 = None
        self.data3 = None

    def test_rotate(self):
        """Test rotate."""
        npt.assert_array_equal(
            np_adjust.rotate(self.data1),
            np.array([[8, 7, 6], [5, 4, 3], [2, 1, 0]]),
            err_msg='Incorrect rotation',
        )

    def test_rotate_stack(self):
        """Test rotate_stack."""
        npt.assert_array_equal(
            np_adjust.rotate_stack(self.data2),
            np.array([
                [[8, 7, 6], [5, 4, 3], [2, 1, 0]],
                [[17, 16, 15], [14, 13, 12], [11, 10, 9]],
            ]),
            err_msg='Incorrect stack rotation',
        )

    def test_pad2d(self):
        """Test pad2d."""
        npt.assert_array_equal(
            np_adjust.pad2d(self.data1, (1, 1)),
            self.data3,
            err_msg='Incorrect padding',
        )

        npt.assert_array_equal(
            np_adjust.pad2d(self.data1, 1),
            self.data3,
            err_msg='Incorrect padding',
        )

        npt.assert_array_equal(
            np_adjust.pad2d(self.data1, np.array([1, 1])),
            self.data3,
            err_msg='Incorrect padding',
        )

        npt.assert_raises(ValueError, np_adjust.pad2d, self.data1, '1')

    def test_fancy_transpose(self):
        """Test fancy_transpose."""
        npt.assert_array_equal(
            np_adjust.fancy_transpose(self.data2),
            np.array([
                [[0, 3, 6], [9, 12, 15]],
                [[1, 4, 7], [10, 13, 16]],
                [[2, 5, 8], [11, 14, 17]],
            ]),
            err_msg='Incorrect fancy transpose',
        )

    def test_ftr(self):
        """Test ftr."""
        npt.assert_array_equal(
            np_adjust.ftr(self.data2),
            np.array([
                [[0, 3, 6], [9, 12, 15]],
                [[1, 4, 7], [10, 13, 16]],
                [[2, 5, 8], [11, 14, 17]],
            ]),
            err_msg='Incorrect fancy transpose: ftr',
        )

    def test_ftl(self):
        """Test ftl."""
        npt.assert_array_equal(
            np_adjust.ftl(self.data2),
            np.array([
                [[0, 9], [1, 10], [2, 11]],
                [[3, 12], [4, 13], [5, 14]],
                [[6, 15], [7, 16], [8, 17]],
            ]),
            err_msg='Incorrect fancy transpose: ftl',
        )


class TransformTestCase(TestCase):
    """Test case for transform module."""

    def setUp(self):
        """Set test parameter values."""
        self.cube = np.arange(16).reshape((4, 2, 2))
        self.map = np.array(
            [[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]],
        )
        self.matrix = np.array(
            [[0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15]],
        )
        self.layout = (2, 2)

    def tearDown(self):
        """Unset test parameter values."""
        self.cube = None
        self.map = None
        self.layout = None

    def test_cube2map(self):
        """Test cube2map."""
        npt.assert_array_equal(
            transform.cube2map(self.cube, self.layout),
            self.map,
            err_msg='Incorrect transformation: cube2map',
        )

        npt.assert_raises(
            ValueError,
            transform.cube2map,
            self.map,
            self.layout,
        )

        npt.assert_raises(ValueError, transform.cube2map, self.cube, (3, 3))

    def test_map2cube(self):
        """Test map2cube."""
        npt.assert_array_equal(
            transform.map2cube(self.map, self.layout),
            self.cube,
            err_msg='Incorrect transformation: map2cube',
        )

        npt.assert_raises(ValueError, transform.map2cube, self.map, (3, 3))

    def test_map2matrix(self):
        """Test map2matrix."""
        npt.assert_array_equal(
            transform.map2matrix(self.map, self.layout),
            self.matrix,
            err_msg='Incorrect transformation: map2matrix',
        )

    def test_matrix2map(self):
        """Test matrix2map."""
        npt.assert_array_equal(
            transform.matrix2map(self.matrix, self.map.shape),
            self.map,
            err_msg='Incorrect transformation: matrix2map',
        )

    def test_cube2matrix(self):
        """Test cube2matrix."""
        npt.assert_array_equal(
            transform.cube2matrix(self.cube),
            self.matrix,
            err_msg='Incorrect transformation: cube2matrix',
        )

    def test_matrix2cube(self):
        """Test matrix2cube."""
        npt.assert_array_equal(
            transform.matrix2cube(self.matrix, self.cube[0].shape),
            self.cube,
            err_msg='Incorrect transformation: matrix2cube',
        )


class TypesTestCase(TestCase):
    """Test case for types module."""

    def setUp(self):
        """Set test parameter values."""
        self.data1 = list(range(5))
        self.data2 = np.arange(5)
        self.data3 = np.arange(5).astype(float)

    def tearDown(self):
        """Unset test parameter values."""
        self.data1 = None
        self.data2 = None
        self.data3 = None

    def test_check_float(self):
        """Test check_float."""
        npt.assert_array_equal(
            types.check_float(1.0),
            1.0,
            err_msg='Float check failed',
        )

        npt.assert_array_equal(
            types.check_float(1),
            1.0,
            err_msg='Float check failed',
        )

        npt.assert_array_equal(
            types.check_float(self.data1),
            self.data3,
            err_msg='Float check failed',
        )

        npt.assert_array_equal(
            types.check_float(self.data2),
            self.data3,
            err_msg='Float check failed',
        )

        npt.assert_raises(TypeError, types.check_float, '1')

    def test_check_int(self):
        """Test check_int."""
        npt.assert_array_equal(
            types.check_int(1),
            1,
            err_msg='Float check failed',
        )

        npt.assert_array_equal(
            types.check_int(1.0),
            1,
            err_msg='Float check failed',
        )

        npt.assert_array_equal(
            types.check_int(self.data1),
            self.data2,
            err_msg='Float check failed',
        )

        npt.assert_array_equal(
            types.check_int(self.data3),
            self.data2,
            err_msg='Int check failed',
        )

        npt.assert_raises(TypeError, types.check_int, '1')

    def test_check_npndarray(self):
        """Test check_npndarray."""
        npt.assert_raises(
            TypeError,
            types.check_npndarray,
            self.data3,
            dtype=np.integer,
        )


class TestBackend(TestCase):
    """Test the backend codes."""

    def setUp(self):
        """Set test parameter values."""
        self.input = np.array([10, 10])

    @skipIf(LIBRARIES['tensorflow'] is None, 'tensorflow library not installed')
    def test_tf_backend(self):
        """Test tensorflow backend."""
        xp, backend = get_backend('tensorflow')
        if backend != ':tensorflow' or xp != LIBRARIES['tensorflow']:
            raise AssertionError('tensorflow get_backend fails!')
        tf_input = change_backend(self.input, 'tensorflow')
        if (
            get_array_module(LIBRARIES['tensorflow'].ones(1)) != LIBRARIES['tensorflow']
            or get_array_module(tf_input) != LIBRARIES['tensorflow']
        ):
            raise AssertionError('tensorflow backend fails!')

    @skipIf(LIBRARIES['cupy'] is None, 'cupy library not installed')
    def test_cp_backend(self):
        """Test cupy backend."""
        xp, backend = get_backend('cupy')
        if backend != 'cupy' or xp != LIBRARIES['cupy']:
            raise AssertionError('cupy get_backend fails!')
        cp_input = change_backend(self.input, 'cupy')
        if (
            get_array_module(LIBRARIES['cupy'].ones(1)) != LIBRARIES['cupy']
            or get_array_module(cp_input) != LIBRARIES['cupy']
        ):
            raise AssertionError('cupy backend fails!')

    def test_np_backend(self):
        """Test numpy backend."""
        xp, backend = get_backend('numpy')
        if backend != 'numpy' or xp != LIBRARIES['numpy']:
            raise AssertionError('numpy get_backend fails!')
        np_input = change_backend(self.input, 'numpy')
        if (
            get_array_module(LIBRARIES['numpy'].ones(1)) != LIBRARIES['numpy']
            or get_array_module(np_input) != LIBRARIES['numpy']
        ):
            raise AssertionError('numpy backend fails!')

    def tearDown(self):
        """Tear Down of objects."""
        self.input = None
