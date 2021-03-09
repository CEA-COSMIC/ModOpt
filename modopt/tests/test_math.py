# -*- coding: utf-8 -*-

"""UNIT TESTS FOR MATH.

This module contains unit tests for the modopt.math module.

:Author: Samuel Farrens <samuel.farrens@cea.fr>

"""

from unittest import TestCase, skipIf, skipUnless
import numpy as np
import numpy.testing as npt
import sys
from modopt.math import *
try:
    import astropy
except ImportError:  # pragma: no cover
    import_astropy = False
else:  # pragma: no cover
    import_astropy = True
try:
    if sys.version_info.minor == 5:
        from skimage.measure import compare_ssim
    elif sys.version_info.minor > 5:
        from skimage.metrics import structural_similarity as compare_ssim
except ImportError:  # pragma: no cover
    import_skimage = False
else:
    import_skimage = True


class ConvolveTestCase(TestCase):

    def setUp(self):

        self.data1 = np.arange(18).reshape(2, 3, 3)
        self.data2 = self.data1 + 1

    def tearDown(self):

        self.data1 = None
        self.data2 = None

    @skipUnless(import_astropy, 'Astropy not installed.')  # pragma: no cover
    def test_convolve_astropy(self):

        npt.assert_allclose(convolve.convolve(self.data1[0], self.data2[0],
                            method='astropy'),
                            np.array([[210., 201., 210.],
                                     [129., 120., 129.],
                                     [210., 201., 210.]]),
                            err_msg='Incorrect convolution: astropy')

        npt.assert_raises(ValueError, convolve.convolve, self.data1[0],
                          self.data2)

        npt.assert_raises(ValueError, convolve.convolve, self.data1[0],
                          self.data2[0], method='bla')

    def test_convolve_scipy(self):

        npt.assert_allclose(convolve.convolve(self.data1[0], self.data2[0],
                            method='scipy'),
                            np.array([[14., 35., 38.], [57., 120., 111.],
                                     [110., 197., 158.]]),
                            err_msg='Incorrect convolution: scipy')

    def test_convolve_stack(self):

        npt.assert_allclose(convolve.convolve_stack(self.data1, self.data2),
                            np.array([[[14., 35., 38.],
                                      [57., 120., 111.],
                                      [110., 197., 158.]],
                                     [[518., 845., 614.],
                                      [975., 1578., 1137.],
                                      [830., 1331., 950.]]]),
                            err_msg='Incorrect convolution: stack')

    def test_convolve_stack_rot(self):

        npt.assert_allclose(convolve.convolve_stack(self.data1, self.data2,
                            rot_kernel=True),
                            np.array([[[66., 115., 82.],
                                      [153., 240., 159.],
                                      [90., 133., 82.]],
                                     [[714., 1087., 730.],
                                      [1125., 1698., 1131.],
                                      [738., 1105., 730.]]]),
                            err_msg='Incorrect convolution: stack rot')


class MatrixTestCase(TestCase):

    def setUp(self):

        self.data1 = np.arange(9).reshape(3, 3)
        self.data2 = np.arange(3)
        self.data3 = np.arange(6).reshape(2, 3)
        np.random.seed(1)
        self.pmInstance1 = matrix.PowerMethod(lambda x: x.dot(x.T),
                                              self.data1.shape, verbose=True)
        np.random.seed(1)
        self.pmInstance2 = matrix.PowerMethod(lambda x: x.dot(x.T),
                                              self.data1.shape, auto_run=False,
                                              verbose=True)
        self.pmInstance2.get_spec_rad(max_iter=1)

    def tearDown(self):

        self.data1 = None
        self.data2 = None

    def test_gram_schmidt_orthonormal(self):

        npt.assert_allclose(matrix.gram_schmidt(self.data1),
                            np.array([[0., 0.4472136, 0.89442719],
                                     [0.91287093, 0.36514837, -0.18257419],
                                     [-1., 0., 0.]]),
                            err_msg='Incorrect Gram-Schmidt: orthonormal')

        npt.assert_raises(ValueError, matrix.gram_schmidt, self.data1,
                          return_opt='bla')

    def test_gram_schmidt_orthogonal(self):

        npt.assert_allclose(matrix.gram_schmidt(self.data1,
                            return_opt='orthogonal'),
                            np.array([[0.00000000e+00, 1.00000000e+00,
                                       2.00000000e+00],
                                      [3.00000000e+00, 1.20000000e+00,
                                       -6.00000000e-01],
                                      [-1.77635684e-15, 0.00000000e+00,
                                       0.00000000e+00]]),
                            err_msg='Incorrect Gram-Schmidt: orthogonal')

    def test_gram_schmidt_both(self):

        npt.assert_allclose(matrix.gram_schmidt(self.data1, return_opt='both'),
                            (np.array([[0.00000000e+00, 1.00000000e+00,
                                        2.00000000e+00],
                                       [3.00000000e+00, 1.20000000e+00,
                                        -6.00000000e-01],
                                       [-1.77635684e-15, 0.00000000e+00,
                                       0.00000000e+00]]),
                             np.array([[0., 0.4472136, 0.89442719],
                                      [0.91287093, 0.36514837, -0.18257419],
                                      [-1., 0., 0.]])),
                            err_msg='Incorrect Gram-Schmidt: both')

    def test_nuclear_norm(self):

        npt.assert_almost_equal(matrix.nuclear_norm(self.data1),
                                15.49193338482967,
                                err_msg='Incorrect nuclear norm')

    def test_project(self):

        npt.assert_array_equal(matrix.project(self.data2, self.data2 + 3),
                               np.array([0., 2.8, 5.6]),
                               err_msg='Incorrect projection')

    def test_rot_matrix(self):

        npt.assert_allclose(matrix.rot_matrix(np.pi / 6),
                            np.array([[0.8660254, -0.5],
                                      [0.5, 0.8660254]]),
                            err_msg='Incorrect rotation matrix')

    def test_rotate(self):

        npt.assert_array_equal(matrix.rotate(self.data1, np.pi / 2),
                               np.array([[2, 5, 8], [1, 4, 7], [0, 3, 6]]),
                               err_msg='Incorrect rotation')

        npt.assert_raises(ValueError, matrix.rotate, self.data3, np.pi / 2)

    def test_PowerMethod_converged(self):

        npt.assert_almost_equal(self.pmInstance1.spec_rad,
                                0.90429242629600837,
                                err_msg='Incorrect spectral radius: converged')

        npt.assert_almost_equal(self.pmInstance1.inv_spec_rad,
                                1.1058369736612865,
                                err_msg='Incorrect inverse spectral radius: '
                                        'converged')

    def test_PowerMethod_unconverged(self):

        npt.assert_almost_equal(self.pmInstance2.spec_rad,
                                0.92048833577059219,
                                err_msg='Incorrect spectral radius: '
                                        'unconverged')

        npt.assert_almost_equal(self.pmInstance2.inv_spec_rad,
                                1.0863798715741946,
                                err_msg='Incorrect inverse spectral radius: '
                                        'unconverged')


class MetricsTestCase(TestCase):

    def setUp(self):

        self.data1 = np.arange(49).reshape(7, 7)
        self.mask = np.ones(self.data1.shape)
        self.ssim_res = 0.8963363560519094
        self.ssim_mask_res = 0.805154442543846
        self.snr_res = 10.134554256920536
        self.psnr_res = 14.860761791850397
        self.mse_res = 0.03265305507330247
        self.nrmse_res = 0.31136678840022625

    def tearDown(self):

        self.data1 = None
        self.mask = None
        self.ssim_res = None
        self.ssim_mask_res = None
        self.psnr_res = None
        self.mse_res = None
        self.nrmse_res = None

    @skipIf(import_skimage, 'skimage is installed.')  # pragma: no cover
    def test_ssim_skimage_error(self):

        npt.assert_raises(ImportError, metrics.ssim, self.data1, self.data1)

    @skipUnless(import_skimage, 'skimage not installed.')  # pragma: no cover
    def test_ssim(self):

        npt.assert_almost_equal(metrics.ssim(self.data1, self.data1 ** 2),
                                self.ssim_res,
                                err_msg='Incorrect SSIM result')

        npt.assert_almost_equal(metrics.ssim(self.data1, self.data1 ** 2,
                                mask=self.mask), self.ssim_mask_res,
                                err_msg='Incorrect SSIM result')

        npt.assert_raises(ValueError, metrics.ssim, self.data1, self.data1,
                          mask=1)

    def test_snr(self):

        npt.assert_almost_equal(metrics.snr(self.data1, self.data1 ** 2),
                                self.snr_res,
                                err_msg='Incorrect SNR result')

        npt.assert_almost_equal(metrics.snr(self.data1, self.data1 ** 2,
                                mask=self.mask), self.snr_res,
                                err_msg='Incorrect SNR result')

    def test_psnr(self):

        npt.assert_almost_equal(metrics.psnr(self.data1, self.data1 ** 2),
                                self.psnr_res,
                                err_msg='Incorrect PSNR result')

        npt.assert_almost_equal(metrics.psnr(self.data1, self.data1 ** 2,
                                mask=self.mask), self.psnr_res,
                                err_msg='Incorrect PSNR result')

    def test_mse(self):

        npt.assert_almost_equal(metrics.mse(self.data1, self.data1 ** 2),
                                self.mse_res,
                                err_msg='Incorrect MSE result')

        npt.assert_almost_equal(metrics.mse(self.data1, self.data1 ** 2,
                                mask=self.mask), self.mse_res,
                                err_msg='Incorrect MSE result')

    def test_nrmse(self):

        npt.assert_almost_equal(metrics.nrmse(self.data1, self.data1 ** 2),
                                self.nrmse_res,
                                err_msg='Incorrect NRMSE result')

        npt.assert_almost_equal(metrics.nrmse(self.data1, self.data1 ** 2,
                                mask=self.mask), self.nrmse_res,
                                err_msg='Incorrect NRMSE result')


class StatsTestCase(TestCase):

    def setUp(self):

        self.data1 = np.arange(9).reshape(3, 3)
        self.data2 = np.arange(18).reshape(2, 3, 3)

    def tearDown(self):

        self.data1 = None

    @skipIf(import_astropy, 'Astropy is installed.')  # pragma: no cover
    def test_gaussian_kernel_astropy_error(self):

        npt.assert_raises(ImportError, stats.gaussian_kernel,
                          self.data1.shape, 1)

    @skipUnless(import_astropy, 'Astropy not installed.')  # pragma: no cover
    def test_gaussian_kernel_max(self):

        npt.assert_allclose(stats.gaussian_kernel(self.data1.shape, 1),
                            np.array([[0.36787944, 0.60653066, 0.36787944],
                                      [0.60653066, 1., 0.60653066],
                                      [0.36787944, 0.60653066, 0.36787944]]),
                            err_msg='Incorrect gaussian kernel: max norm')

        npt.assert_raises(ValueError, stats.gaussian_kernel,
                          self.data1.shape, 1, norm='bla')

    @skipUnless(import_astropy, 'Astropy not installed.')  # pragma: no cover
    def test_gaussian_kernel_sum(self):

        npt.assert_allclose(stats.gaussian_kernel(self.data1.shape, 1,
                            norm='sum'),
                            np.array([[0.07511361, 0.1238414, 0.07511361],
                                      [0.1238414, 0.20417996, 0.1238414],
                                      [0.07511361, 0.1238414, 0.07511361]]),
                            err_msg='Incorrect gaussian kernel: sum norm')

    @skipUnless(import_astropy, 'Astropy not installed.')  # pragma: no cover
    def test_gaussian_kernel_none(self):

        npt.assert_allclose(stats.gaussian_kernel(self.data1.shape, 1,
                            norm='none'),
                            np.array([[0.05854983, 0.09653235, 0.05854983],
                                      [0.09653235, 0.15915494, 0.09653235],
                                      [0.05854983, 0.09653235, 0.05854983]]),
                            err_msg='Incorrect gaussian kernel: sum norm')

    def test_mad(self):

        npt.assert_equal(stats.mad(self.data1), 2.0,
                         err_msg='Incorrect median absolute deviation')

    def test_mse(self):

        npt.assert_equal(stats.mse(self.data1, self.data1 + 2), 4.0,
                         err_msg='Incorrect mean squared error')

    def test_psnr_starck(self):

        npt.assert_almost_equal(stats.psnr(self.data1, self.data1 + 2),
                                12.041199826559248,
                                err_msg='Incorrect PSNR: starck')

        npt.assert_raises(ValueError, stats.psnr, self.data1, self.data1,
                          method='bla')

    def test_psnr_wiki(self):

        npt.assert_almost_equal(stats.psnr(self.data1, self.data1 + 2,
                                method='wiki'),
                                42.110203695399477,
                                err_msg='Incorrect PSNR: wiki')

    def test_psnr_stack(self):

        npt.assert_almost_equal(stats.psnr_stack(self.data2, self.data2 + 2),
                                12.041199826559248,
                                err_msg='Incorrect PSNR stack')

        npt.assert_raises(ValueError, stats.psnr_stack, self.data1, self.data1)

    def test_sigma_mad(self):

        npt.assert_almost_equal(stats.sigma_mad(self.data1),
                                2.9651999999999998,
                                err_msg='Incorrect sigma from MAD')
