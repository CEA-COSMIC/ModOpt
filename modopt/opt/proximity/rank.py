"""Matrix Based Proximity Operators."""

from modopt.base.transform import cube2matrix, matrix2cube
from modopt.math.matrix import nuclear_norm
from modopt.opt.proximity.base import ProximityParent
from modopt.signal.svd import svd_thresh, svd_thresh_coef, svd_thresh_coef_fast


class LowRankMatrix(ProximityParent):
    """Low-rank Proximity Operator.

    This class defines the low-rank proximity operator.

    Parameters
    ----------
    threshold : float
        Threshold value
    treshold_type : {'hard', 'soft'}
        Threshold type (options are 'hard' or 'soft', default is 'soft')
    lowr_type : {'standard', 'ngole'}
        Low-rank implementation (options are 'standard' or 'ngole', default is
        'standard')
    initial_rank: int, optional
        Initial guess of the rank of future input_data.
        If provided this will save computation time.
    operator : class
        Operator class ('ngole' only)

    Examples
    --------
    >>> import numpy as np
    >>> from modopt.opt.proximity import LowRankMatrix
    >>> a = np.arange(9).reshape(3, 3).astype(float)
    >>> inst = LowRankMatrix(10.0, thresh_type='hard')
    >>> inst.op(a)
    array([[0.89642146, 1.0976143 , 1.29880715],
           [3.29284291, 4.03188864, 4.77093436],
           [5.68926437, 6.96616297, 8.24306156]])
    >>> inst.cost(a, verbose=True)
     - NUCLEAR NORM (X): 154.91933384829667
    154.91933384829667

    See Also
    --------
    ProximityParent : parent class
    modopt.signal.svd.svd_thresh : SVD thresholding function
    modopt.signal.svd.svd_thresh_coef : SVD coefficient thresholding function
    modopt.math.matrix.nuclear_norm : nuclear norm implementation

    """

    def __init__(
        self,
        threshold,
        thresh_type='soft',
        lowr_type='standard',
        initial_rank=None,
        operator=None,
    ):

        self.thresh = threshold
        self.thresh_type = thresh_type
        self.lowr_type = lowr_type
        self.operator = operator
        self.op = self._op_method
        self.cost = self._cost_method
        self.rank = initial_rank

    def _op_method(self, input_data, extra_factor=1.0, rank=None):
        """Operator.

        This method returns the input data after the singular values have been
        thresholded.

        Parameters
        ----------
        input_data : numpy.ndarray
            Input data array
        extra_factor : float
            Additional multiplication factor (default is ``1.0``)
        rank: int, optional
            Estimation of the rank to save computation time in standard mode,
            if not set an internal estimation is used.

        Returns
        -------
        numpy.ndarray
            SVD thresholded data

        Raises
        ------
        ValueError
            if lowr_type is not in ``{'standard', 'ngole'}``
        """
        # Update threshold with extra factor.
        threshold = self.thresh * extra_factor
        if self.lowr_type == 'standard' and self.rank is None and rank is None:
            data_matrix = svd_thresh(
                cube2matrix(input_data),
                threshold,
                thresh_type=self.thresh_type,
            )
        elif self.lowr_type == 'standard':
            data_matrix, update_rank = svd_thresh_coef_fast(
                cube2matrix(input_data),
                threshold,
                n_vals=rank or self.rank,
                extra_vals=5,
                thresh_type=self.thresh_type,
            )
            self.rank = update_rank  # save for future use

        elif self.lowr_type == 'ngole':
            data_matrix = svd_thresh_coef(
                cube2matrix(input_data),
                self.operator,
                threshold,
                thresh_type=self.thresh_type,
            )
        else:
            raise ValueError('lowr_type should be standard or ngole')

        # Return updated data.
        return matrix2cube(data_matrix, input_data.shape[1:])

    def _cost_method(self, *args, **kwargs):
        """Calculate low-rank component of the cost.

        This method returns the nuclear norm error of the deconvolved data in
        matrix form.

        Parameters
        ----------
        *args : tuple
            Positional arguments
        **kwargs : dict
            Keyword arguments

        Returns
        -------
        float
            Low-rank cost component

        """
        cost_val = self.thresh * nuclear_norm(cube2matrix(args[0]))

        if 'verbose' in kwargs and kwargs['verbose']:
            print(' - NUCLEAR NORM (X):', cost_val)

        return cost_val
