"""Base classes for linear operators."""

import numpy as np

from modopt.base.types import check_callable
from modopt.base.backend import get_array_module


class LinearParent:
    """Linear Operator Parent Class.

    This class sets the structure for defining linear operator instances.

    Parameters
    ----------
    op : callable
        Callable function that implements the linear operation
    adj_op : callable
        Callable function that implements the linear adjoint operation

    Examples
    --------
    >>> from modopt.opt.linear import LinearParent
    >>> a = LinearParent(lambda x: x * 2, lambda x: x ** 3)
    >>> a.op(2)
    4
    >>> a.adj_op(2)
    8

    """

    def __init__(self, op, adj_op):
        self.op = op
        self.adj_op = adj_op

    @property
    def op(self):
        """Linear Operator."""
        return self._op

    @op.setter
    def op(self, operator):
        self._op = check_callable(operator)

    @property
    def adj_op(self):
        """Linear Adjoint Operator."""
        return self._adj_op

    @adj_op.setter
    def adj_op(self, operator):
        self._adj_op = check_callable(operator)


class Identity(LinearParent):
    """Identity Operator Class.

    This is a dummy class that can be used in the optimisation classes.

    See Also
    --------
    LinearParent : parent class

    """

    def __init__(self):
        self.op = lambda input_data: input_data
        self.adj_op = self.op
        self.cost = lambda *args, **kwargs: 0


class MatrixOperator(LinearParent):
    """
    Matrix Operator class.

    This class transforms an array into a suitable linear operator.
    """

    def __init__(self, array):
        self.op = lambda x: array @ x
        xp = get_array_module(array)

        if xp.any(xp.iscomplex(array)):
            self.adj_op = lambda x: array.T.conjugate() @ x
        else:
            self.adj_op = lambda x: array.T @ x


class LinearCombo(LinearParent):
    """Linear Combination Class.

    This class defines a combination of linear transform operators.

    Parameters
    ----------
    operators : list, tuple or numpy.ndarray
        List of linear operator class instances
    weights : list, tuple or numpy.ndarray, optional
        List of weights for combining the linear adjoint operator results

    Examples
    --------
    >>> from modopt.opt.linear import LinearCombo, LinearParent
    >>> a = LinearParent(lambda x: x * 2, lambda x: x ** 3)
    >>> b = LinearParent(lambda x: x * 4, lambda x: x ** 5)
    >>> c = LinearCombo([a, b])
    >>> a.op(2)
    4
    >>> b.op(2)
    8
    >>> c.op(2)
    array([4, 8], dtype=object)
    >>> a.adj_op(2)
    8
    >>> b.adj_op(2)
    32
    >>> c.adj_op([2, 2])
    20.0

    See Also
    --------
    LinearParent : parent class
    """

    def __init__(self, operators, weights=None):
        operators, weights = self._check_inputs(operators, weights)
        self.operators = operators
        self.weights = weights
        self.op = self._op_method
        self.adj_op = self._adj_op_method

    def _check_type(self, input_val):
        """Check input type.

        This method checks if the input is a list, tuple or a numpy array and
        converts the input to a numpy array.

        Parameters
        ----------
        input_val : any
            Any input object

        Returns
        -------
        numpy.ndarray
            Numpy array of inputs

        Raises
        ------
        TypeError
            For invalid input type
        ValueError
            If input list is empty

        """
        if not isinstance(input_val, (list, tuple, np.ndarray)):
            raise TypeError(
                "Invalid input type, input must be a list, tuple or numpy " + "array.",
            )

        input_val = np.array(input_val)

        if not input_val.size:
            raise ValueError("Input list is empty.")

        return input_val

    def _check_inputs(self, operators, weights):
        """Check inputs.

        This method cheks that the input operators and weights are correctly
        formatted.

        Parameters
        ----------
        operators : list, tuple or numpy.ndarray
            List of linear operator class instances
        weights : list, tuple or numpy.ndarray
            List of weights for combining the linear adjoint operator results

        Returns
        -------
        tuple
            Operators and weights

        Raises
        ------
        ValueError
            If the number of weights does not match the number of operators
        TypeError
            If the individual weight values are not floats

        """
        operators = self._check_type(operators)

        for operator in operators:
            if not hasattr(operator, "op"):
                raise ValueError('Operators must contain "op" method.')

            if not hasattr(operator, "adj_op"):
                raise ValueError('Operators must contain "adj_op" method.')

            operator.op = check_callable(operator.op)
            operator.adj_op = check_callable(operator.adj_op)

        if not isinstance(weights, type(None)):
            weights = self._check_type(weights)

            if weights.size != operators.size:
                raise ValueError(
                    "The number of weights must match the number of " + "operators.",
                )

            if not np.issubdtype(weights.dtype, np.floating):
                raise TypeError("The weights must be a list of float values.")

        return operators, weights

    def _op_method(self, input_data):
        """Operator.

        This method returns the input data operated on by all of the operators.

        Parameters
        ----------
        input_data : numpy.ndarray
            Input data array

        Returns
        -------
        numpy.ndarray
            Linear operation results

        """
        res = np.empty(len(self.operators), dtype=np.ndarray)

        for index, _ in enumerate(self.operators):
            res[index] = self.operators[index].op(input_data)

        return res

    def _adj_op_method(self, input_data):
        """Adjoint operator.

        This method returns the combination of the result of all of the
        adjoint operators. If weights are provided the comibination is the sum
        of the weighted results, otherwise the combination is the mean.

        Parameters
        ----------
        input_data : numpy.ndarray
            Input data array

        Returns
        -------
        numpy.ndarray
            Adjoint operation results

        """
        if isinstance(self.weights, type(None)):
            return np.mean(
                [
                    operator.adj_op(elem)
                    for elem, operator in zip(input_data, self.operators)
                ],
                axis=0,
            )

        return np.sum(
            [
                weight * operator.adj_op(elem)
                for elem, operator, weight in zip(
                    input_data,
                    self.operators,
                    self.weights,
                )
            ],
            axis=0,
        )
