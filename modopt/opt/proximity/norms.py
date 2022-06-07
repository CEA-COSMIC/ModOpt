"""Vector Norm-based Proximal Operators."""
import numpy as np

from modopt.opt.proximity.base import ProximityParent

try:
    from sklearn.isotonic import isotonic_regression
except ImportError:
    import_sklearn = False
else:
    import_sklearn = True

from modopt.interface.errors import warn
from modopt.signal.noise import thresh


class SparseThreshold(ProximityParent):
    """Sparse Threshold Proximity Operator.

    This class defines the sparse thresholding proximity operator.

    Parameters
    ----------
    linear : class
        Linear operator class
    weights : numpy.ndarray
        Input array of weights
    thresh_type : {'hard', 'soft'}, optional
        Threshold type (default is 'soft')

    See Also
    --------
    ProximityParent : parent class
    modopt.signal.noise.thresh : thresholding function

    """

    def __init__(self, linear, weights, thresh_type='soft'):

        self._linear = linear
        self.weights = weights
        self._thresh_type = thresh_type
        self.op = self._op_method
        self.cost = self._cost_method

    def _op_method(self, input_data, extra_factor=1.0):
        """Operator.

        This method returns the input data thresholded by the weights.

        Parameters
        ----------
        input_data : numpy.ndarray
            Input data array
        extra_factor : float
            Additional multiplication factor (default is ``1.0``)

        Returns
        -------
        numpy.ndarray
            Thresholded data

        """
        threshold = self.weights * extra_factor

        return thresh(input_data, threshold, self._thresh_type)

    def _cost_method(self, *args, **kwargs):
        """Calculate sparsity component of the cost.

        This method returns the l1 norm error of the weighted wavelet
        coefficients.

        Parameters
        ----------
        *args : tuple
            Positional arguments
        **kwargs : dict
            Keyword arguments

        Returns
        -------
        float
            Sparsity cost component

        """
        cost_val = np.sum(np.abs(self.weights * self._linear.op(args[0])))

        if 'verbose' in kwargs and kwargs['verbose']:
            print(' - L1 NORM (X):', cost_val)

        return cost_val


class OrderedWeightedL1Norm(ProximityParent):
    """Ordered Weighted L1 norm proximity operator.

    This class defines the OWL proximity operator described in
    :cite:`figueiredo2014`.

    Parameters
    ----------
    weights : numpy.ndarray
        Weights values they should be sorted in a non-increasing order

    Examples
    --------
    >>> import numpy as np
    >>> from modopt.opt.proximity import OrderedWeightedL1Norm
    >>> A = np.arange(5) * 5
    >>> A
    array([ 0,  5, 10, 15, 20])
    >>> weights = np.arange(5)[::-1]
    >>> prox_op = OrderedWeightedL1Norm(weights)
    >>> prox_op.weights
    array([4, 3, 2, 1, 0])
    >>> prox_op.op(A)
    array([ 0.,  4.,  8., 12., 16.])
    >>> prox_op.cost(A, verbose=True)
     - OWL NORM (X): 50
    50

    See Also
    --------
    ProximityParent : parent class
    sklearn.isotonic.isotonic_regression : isotonic regression implementation

    """

    def __init__(self, weights):
        if not import_sklearn:  # pragma: no cover
            raise ImportError(
                'Required version of Scikit-Learn package not found see '
                + 'documentation for details: '
                + 'https://cea-cosmic.github.io/ModOpt/#optional-packages',
            )
        if np.max(np.diff(weights)) > 0:
            raise ValueError('Weights must be non increasing')
        self.weights = weights.flatten()
        if (self.weights < 0).any():
            raise ValueError(
                'The weight values must be provided in descending order',
            )
        self.op = self._op_method
        self.cost = self._cost_method

    def _op_method(self, input_data, extra_factor=1.0):
        """Operator.

        This method returns the input data after the a clustering and a
        thresholding. Implements (Eq 24) in :cite:`figueiredo2014`.

        Parameters
        ----------
        input_data : numpy.ndarray
            Input data array
        extra_factor : float
            Additional multiplication factor (default is ``1.0``)

        Returns
        -------
        numpy.ndarray
            Thresholded data

        """
        # Update threshold with extra factor.
        threshold = self.weights * extra_factor

        # Squeezing the data
        data_squeezed = np.squeeze(input_data)

        # Sorting (non increasing order) input vector's absolute values
        data_abs = np.abs(data_squeezed)
        data_abs_sort_idx = np.argsort(data_abs)[::-1]
        data_abs = data_abs[data_abs_sort_idx]

        # Projection onto the monotone non-negative cone using
        # isotonic_regression
        data_abs = isotonic_regression(
            data_abs - threshold, y_min=0, increasing=False,
        )

        # Unsorting the data
        data_abs_unsorted = np.empty_like(data_abs)
        data_abs_unsorted[data_abs_sort_idx] = data_abs

        # Putting the sign back
        with np.errstate(invalid='ignore'):
            sign_data = data_squeezed / np.abs(data_squeezed)

        # Removing NAN caused by the sign
        sign_data[np.isnan(sign_data)] = 0

        return np.reshape(sign_data * data_abs_unsorted, input_data.shape)

    def _cost_method(self, *args, **kwargs):
        """Calculate OWL component of the cost.

        This method returns the ordered weighted l1 norm of the data.

        Parameters
        ----------
        *args : tuple
            Positional arguments
        **kwargs : dict
            Keyword arguments

        Returns
        -------
        float
            OWL cost component

        """
        cost_val = np.sum(
            self.weights * np.sort(np.squeeze(np.abs(args[0]))[::-1]),
        )

        if 'verbose' in kwargs and kwargs['verbose']:
            print(' - OWL NORM (X):', cost_val)

        return cost_val


class Ridge(ProximityParent):
    r"""L2-norm Proximity Operator (`i.e.` Shrinkage).

    This class defines the L2-norm proximity operator.

    Parameters
    ----------
    linear : class
        Linear operator class
    weights : numpy.ndarray
        Input array of weights

    Notes
    -----
    Implements the following equation:

    .. math::
        prox(y) = \underset{x \in \mathbb{C}^N}{argmin} 0.5 \|x-y\||_2^2 +
        \alpha\|x\|_2^2

    See Also
    --------
    ProximityParent : parent class

    """

    def __init__(self, linear, weights, thresh_type='soft'):

        self._linear = linear
        self.weights = weights
        self.op = self._op_method
        self.cost = self._cost_method

    def _op_method(self, input_data, extra_factor=1.0):
        """Operator Method.

        This method returns the input data shrinked by the weights

        Parameters
        ----------
        input_data : numpy.ndarray
            Input data array
        extra_factor : float
            Additional multiplication factor (default is ``1.0``)

        Returns
        -------
        numpy.ndarray
            Thresholded data

        """
        threshold = self.weights * extra_factor * 2

        return self._linear.op(input_data) / (1 + threshold)

    def _cost_method(self, *args, **kwargs):
        """Calculate Ridge component of the cost.

        This method returns the l2 norm error of the weighted wavelet
        coefficients.

        Parameters
        ----------
        *args : tuple
            Positional arguments
        **kwargs : dict
            Keyword arguments

        Returns
        -------
        float
            Sparsity cost component

        """
        cost_val = np.sum(
            np.abs(self.weights * self._linear.op(args[0]) ** 2),
        )

        if 'verbose' in kwargs and kwargs['verbose']:
            print(' - L2 NORM (X):', cost_val)

        return cost_val


class ElasticNet(ProximityParent):
    r"""Elastic Net.

    This class defines the Elastic net proximity operator, which is  a
    linear combination between L2 and L1 norm proximity operators,
    described in :cite:`zou2005`.

    Parameters
    ----------
    alpha : numpy.ndarray
        Weights for the L2 norm
    beta : numpy.ndarray
        Weights for the L1 norm

    Notes
    -----
    Implements the following equation:

    .. math::
        prox(y) = \underset{x \in \mathbb{C}^N}{argmin} 0.5 \|x-y\||_2^2 +
        \alpha\|x\|_2^2 + beta*||x||_1

    See Also
    --------
    ProximityParent : parent class
    modopt.signal.noise.thresh : thresholding function

    """

    def __init__(self, linear, alpha, beta):

        self._linear = linear
        self.alpha = alpha
        self.beta = beta
        self.op = self._op_method
        self.cost = self._cost_method

    def _op_method(self, input_data, extra_factor=1.0):
        """Operator.

        This method returns the input data shrinked by the weights.

        Parameters
        ----------
        input_data : numpy.ndarray
            Input data array
        extra_factor : float
            Additional multiplication factor

        Returns
        -------
        numpy.ndarray
            Thresholded data

        """
        soft_threshold = self.beta * extra_factor
        normalization = (self.alpha * 2 * extra_factor + 1)
        return thresh(input_data, soft_threshold, 'soft') / normalization

    def _cost_method(self, *args, **kwargs):
        """Calculate Ridge component of the cost.

        This method returns the l2 norm error of the weighted wavelet
        coefficients.

        Parameters
        ----------
        *args : tuple
            Positional arguments
        **kwargs : dict
            Keyword arguments

        Returns
        -------
        float
            Sparsity cost component

        """
        cost_val = np.sum(
            np.abs(self.alpha * self._linear.op(args[0]) ** 2)
            + np.abs(self.beta * self._linear.op(args[0])),
        )

        if 'verbose' in kwargs and kwargs['verbose']:
            print(' - ELASTIC NET (X):', cost_val)

        return cost_val


class KSupportNorm(ProximityParent):
    """K-support Norm Proximity Operator.

    This class defines the squarred :math:`k`-support norm proximity operator
    described in :cite:`mcdonald2014`.

    Parameters
    ----------
    thresh : float
        Threshold value
    k_value : int
        Hyper-parameter of the :math:`k`-support norm, equivalent to the
        cardinality value for the overlapping group lasso. :math:`k` should be
        included in {1, ..., dim(input_vector)}.

    Notes
    -----
    The :math:`k`-support norm can be seen as an extension to the group-LASSO
    with overlaps with groups of cardianlity at most equal to :math:`k`.
    When :math:`k = 1` the norm is equivalent to the L1-norm.
    When :math:`k` = dimension of the input vector than the norm is equivalent
    to the L2-norm.

    The dual of this norm corresponds to the sum of the k biggest input
    entries.

    Examples
    --------
    >>> import numpy as np
    >>> from modopt.opt.proximity import KSupportNorm
    >>> A = np.arange(5) * 5
    >>> A
    array([ 0,  5, 10, 15, 20])
    >>> prox_op = KSupportNorm(beta=3, k_value=1)
    >>> prox_op.op(A)
    array([0., 0., 0., 0., 5.])
    >>> prox_op.cost(A, verbose=True)
     - K-SUPPORT NORM (X): 7500.0
    7500.0

    See Also
    --------
    ProximityParent : parent class

    """

    def __init__(self, beta, k_value):
        self.beta = beta
        self.k_value = k_value
        self.op = self._op_method
        self.cost = self._cost_method

    @property
    def k_value(self):
        """Get the :math:`k` value."""
        return self._k_value

    @k_value.setter
    def k_value(self, k_val):
        if k_val < 1:
            raise ValueError(
                'The k parameter should be greater or equal than 1',
            )
        self._k_value = k_val

    def _compute_theta(self, input_data, alpha, extra_factor=1.0):
        r"""Compute theta.

        This method computes theta from Corollary 16:

        .. math::
            \begin{align*}
            \theta_i &=
            \begin{cases}
            1, & \text{if} \, \alpha \vert w_i \vert - 2 \lambda > 1\\
            \alpha \vert w_i \vert - 2 \lambda, & \text{if} \
            1 \geq \alpha \vert w_i \vert -2 \lambda \geq 0 \\
            0, & \text{if} \, 0 > \alpha \vert w_i \vert - 2 \lambda
            \end{cases}
            \end{align*}

        Parameters
        ----------
        input_data: numpy.ndarray
            Input data
        alpha: float
            Parameter choosen such that :math:`\sum\theta_i = k`
        extra_factor: float
            Potential extra factor comming from the optimization process
            (default is ``1.0``)

        Returns
        -------
        theta: numpy.ndarray
            Same size as :math:`w` and each component is equal to
            :math:`theta_i`

        """
        alpha_input = np.dot(
            np.expand_dims(alpha, -1),
            np.expand_dims(np.abs(input_data), -1).T,
        )
        theta = np.zeros(alpha_input.shape)
        alpha_beta = alpha_input - self.beta * extra_factor
        theta = alpha_beta * np.logical_and(
            (alpha_beta <= 1),
            (alpha_beta >= 0),
        )
        theta = np.nan_to_num(theta)
        theta += (alpha_input > (self.beta * extra_factor + 1))
        return theta

    def _interpolate(self, alpha0, alpha1, sum0, sum1):
        r"""Linear interpolation of alpha (:math:`\alpha`).

        This method estimats :math:`\alpha^*` such that
        :math:`\sum\theta(\alpha^*)=k` via a linear interpolation.

        Parameters
        ----------
        alpha0: float
            A value for wich :math:`\sum\theta(\alpha^0) \leq k`
        alpha1: float
            A value for which :math:`\sum\theta(\alpha^1) \leq k`
        sum0: float
            Value of :math:`\sum\theta(\alpha^0)`
        sum1: float
            Value of :math:`\sum\theta(\alpha^1)`

        Returns
        -------
        float
            An interpolation for which :math:`\sum\theta(\alpha^*) = k`

        """
        if sum0 == self._k_value:
            return alpha0

        elif sum1 == self._k_value:
            return alpha1

        slope = (sum1 - sum0) / (alpha1 - alpha0)
        b_val = sum0 - slope * alpha0

        return (self._k_value - b_val) / slope

    def _binary_search(self, input_data, alpha, extra_factor=1.0):
        r"""Binary search method.

        This method finds the coordinate of :math:`\alpha^i` such that
        :math:`\sum\theta(\alpha^i) =< k` and
        :math:`\sum\theta(\alpha^{i+1}) >= k` via a binary search method.

        Parameters
        ----------
        input_data: numpy.ndarray
            Absolute value of the input data
        alpha: numpy.ndarray
            Array same size as the input data
        extra_factor: float
            Potential extra factor comming from the optimization process
            (default is ``1.0``)

        Raises
        ------
        ValueError
            For invalid output alpha value

        Returns
        -------
        tuple
            The index where: :math:`\sum\theta(\alpha^i) <= k` and
            :math:`\sum\theta(\alpha^{i+1}) >= k`, The alpha value for which
            :math:`\sum\theta(\alpha^i) <= k`,  The alpha value for which
            :math:`\sum\theta(\alpha^{i+1}) >= k`, Value of
            :math:`\sum\theta(\alpha^i)`,
            Value of :math:`\sum\theta(\alpha^{i + 1})`

        """
        first_idx = 0
        data_abs = np.abs(input_data)
        last_idx = alpha.shape[0] - 1
        found = False
        prev_midpoint = 0
        cnt = 0  # Avoid infinite looops
        tolerance = 1e-4

        # Checking particular to be sure that the solution is in the array
        sum0 = self._compute_theta(data_abs, alpha[0], extra_factor).sum()
        sum1 = self._compute_theta(data_abs, alpha[-1], extra_factor).sum()

        if sum1 <= self._k_value:
            midpoint = alpha.shape[0] - 2
            found = True

        if sum0 >= self._k_value:
            found = True
            midpoint = 0

        while (first_idx <= last_idx) and not found and (cnt < alpha.shape[0]):

            midpoint = (first_idx + last_idx) // 2
            cnt += 1

            if prev_midpoint == midpoint:

                # Particular case
                sum0 = self._compute_theta(
                    data_abs,
                    alpha[first_idx],
                    extra_factor,
                ).sum()
                sum1 = self._compute_theta(
                    data_abs,
                    alpha[last_idx],
                    extra_factor,
                ).sum()

                if (np.abs(sum0 - self._k_value) <= tolerance):
                    found = True
                    midpoint = first_idx

                if (np.abs(sum1 - self._k_value) <= tolerance):
                    found = True
                    midpoint = last_idx - 1
                    # -1 because output is index such that
                    # `sum(theta(alpha[index])) <= k`

                if (first_idx - last_idx) in {1, 2}:
                    sum0 = self._compute_theta(
                        data_abs,
                        alpha[first_idx],
                        extra_factor,
                    ).sum()
                    sum1 = self._compute_theta(
                        data_abs,
                        alpha[last_idx],
                        extra_factor,
                    ).sum()

                    found = sum0 <= self._k_value <= sum1
            sum0 = self._compute_theta(
                data_abs,
                alpha[midpoint],
                extra_factor,
            ).sum()
            sum1 = self._compute_theta(
                data_abs,
                alpha[midpoint + 1],
                extra_factor,
            ).sum()

            found = sum0 <= self._k_value <= sum1

            if sum1 < self._k_value:
                first_idx = midpoint

            elif sum0 > self._k_value:
                last_idx = midpoint

            prev_midpoint = midpoint

        if found:
            return (
                midpoint, alpha[midpoint], alpha[midpoint + 1], sum0, sum1,
            )

        raise ValueError(
            'Cannot find the coordinate of alpha (i) such '
            + 'that sum(theta(alpha[i])) =< k and '
            + 'sum(theta(alpha[i+1])) >= k ',
        )

    def _find_alpha(self, input_data, extra_factor=1.0):
        """Find alpha value to compute theta.

        This method aim at finding alpha such that sum(theta(alpha)) = k.

        Parameters
        ----------
        input_data: numpy.ndarray
            Input data
        extra_factor: float
            Potential extra factor for the weights (default is ``1.0``)

        Returns
        -------
        float
            An interpolation of alpha such that sum(theta(alpha)) = k

        """
        data_size = input_data.shape[0]

        # Computes the alpha^i points line 1 in Algorithm 1.
        alpha = np.ones((data_size * 2)) * self.beta * extra_factor
        data_abs = np.abs(input_data)
        alpha[:data_size] = (
            (self.beta * extra_factor)
            / (data_abs + np.finfo(np.float64).eps)
        )
        alpha[data_size:] = (
            (self.beta * extra_factor + 1)
            / (data_abs + np.finfo(np.float64).eps)
        )
        alpha = np.sort(np.unique(alpha))

        # Identify points alpha^i and alpha^{i+1} line 2. Algorithm 1
        _, *alpha_sum = self._binary_search(
            input_data,
            alpha,
            extra_factor,
        )

        # Interpolate alpha^\star such that its sum is equal to k
        return self._interpolate(*alpha_sum)

    def _op_method(self, input_data, extra_factor=1.0):
        """Operator.

        This method returns the proximity operator of the squared k-support
        norm. Implements (Alg. 1) in :cite:`mcdonald2014`.

        Parameters
        ----------
        input_data : numpy.ndarray
            Input data array
        extra_factor : float
            Additional multiplication factor (default is ``1.0``)

        Returns
        -------
        numpy.ndarray
            Proximal map

        """
        data_shape = input_data.shape
        k_max = np.prod(data_shape)
        if self._k_value > k_max:
            warn(
                'K value of the K-support norm is greater than the input '
                + 'dimension, its value will be set to {0}'.format(k_max),
            )
            self._k_value = k_max

        # Computes line 1., 2. and 3. in Algorithm 1
        alpha = self._find_alpha(np.abs(input_data.flatten()), extra_factor)

        # Computes line 4. in Algorithm 1
        theta = self._compute_theta(np.abs(input_data.flatten()), alpha)

        # Computes line 5. in Algorithm 1.
        rslt = np.nan_to_num(
            (input_data.flatten() * theta)
            / (theta + self.beta * extra_factor),
        )
        return rslt.reshape(data_shape)

    def _find_q(self, sorted_data):
        r"""Find :math:`q`.

        Find the :math:`q` index value.

        Parameters
        ----------
        sorted_data : numpy.ndarray
            Absolute value of the input data sorted in a non-decreasing order

        Returns
        -------
        int
            The :math:`q` index value

        Notes
        -----
        This method finds the value of :math:`q` such that:

        .. math::

            |w_q| \geq \frac{\sum_{j=q+1}^d |w_j|}{k - q} \geq |w_{q+1}|

        where :math:`w` is the input ``sorted_data`` and :math:`k \leq d`.

        """
        first_idx = 0
        last_idx = self._k_value - 1
        found = False
        q_val = (first_idx + last_idx) // 2
        cnt = 0

        # Particular case
        if (sorted_data.sum() / (self._k_value)) >= sorted_data[0]:
            found = True
            q_val = 0

        elif (
            (sorted_data[self._k_value - 1:].sum())
            <= sorted_data[self._k_value - 1]
        ):
            found = True
            q_val = self._k_value - 1

        while (
            not found and cnt < self._k_value
            and (first_idx <= last_idx < self._k_value)
        ):

            q_val = (first_idx + last_idx) // 2
            cnt += 1
            l1_part = sorted_data[q_val:].sum() / (self._k_value - q_val)

            if (
                sorted_data[q_val + 1] <= l1_part <= sorted_data[q_val]
            ):
                found = True

            else:
                if sorted_data[q_val] <= l1_part:
                    last_idx = q_val
                if l1_part <= sorted_data[q_val + 1]:
                    first_idx = q_val

        return q_val

    def _cost_method(self, *args, **kwargs):
        """Calculate :math:`k`-support component of the cost.

        This method returns the :math:`k`-support contribution to the total
        cost.

        Parameters
        ----------
        *args : tuple
            Positional arguments
        **kwargs : dict
            Keyword arguments

        Returns
        -------
        float
            The :math:`k`-support cost component

        """
        data_abs = np.abs(args[0].flatten())
        ix = np.argsort(data_abs)[::-1]
        data_abs = data_abs[ix]  # Sorted absolute value of the data
        q_val = self._find_q(data_abs)
        cost_val = (
            (
                np.sum(data_abs[:q_val] ** 2) * 0.5
                + np.sum(data_abs[q_val:]) ** 2
                / (self._k_value - q_val)
            ) * self.beta
        )

        if 'verbose' in kwargs and kwargs['verbose']:
            print(' - K-SUPPORT NORM (X):', cost_val)

        return cost_val


class GroupLASSO(ProximityParent):
    """Group LASSO norm proximity.

    This class implements the proximity operator of the group-lasso
    regularization as defined in :cite:`yuan2006`, with groups dimension
    being the first dimension.

    Parameters
    ----------
    weights : numpy.ndarray
        Input array of weights

    Examples
    --------
    >>> import numpy as np
    >>> from modopt.opt.proximity import GroupLASSO
    >>> A = np.arange(15).reshape(3, 5)
    >>> A
    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14]])
    >>> prox_op = GroupLASSO(weights=3)
    >>> prox_op.op(A)
    array([[ 0.        ,  0.76133281,  1.5725177 ,  2.42145809,  3.29895251],
           [ 3.65835921,  4.56799689,  5.50381195,  6.45722157,  7.42264316],
           [ 7.31671843,  8.37466096,  9.4351062 , 10.49298505, 11.5463338 ]])
    >>> prox_op.cost(A, verbose=True)
    211.37821733946427

    See Also
    --------
    ProximityParent : parent class

    """

    def __init__(self, weights):
        self.weights = weights
        self.op = self._op_method
        self.cost = self._cost_method

    def _op_method(self, input_data, extra_factor=1.0):
        """Operator.

        This method returns the input data thresholded by the weights.

        Parameters
        ----------
        input_data : numpy.ndarray
            Input data array
        extra_factor : float
            Additional multiplication factor (default is ``1.0``)

        Returns
        -------
        numpy.ndarray
            With proximal of GroupLASSO regularization

        """
        norm2 = np.linalg.norm(input_data, axis=0)
        denominator = np.maximum(norm2, np.finfo(np.float32).eps)

        return input_data * np.maximum(
            0,
            (1.0 - self.weights * extra_factor / denominator),
        )

    def _cost_method(self, input_data):
        """Calculate the group LASSO component of the cost.

        This method calculate the cost function of the proximable part.

        Parameters
        ----------
        input_data : numpy.ndarray
            Input array of the sparse code

        Returns
        -------
        float
            The cost of GroupLASSO regularizer

        """
        return np.sum(self.weights * np.linalg.norm(input_data, axis=0))
