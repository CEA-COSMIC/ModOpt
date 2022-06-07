"""
Base Proximity operators.

This module contains basic class to create and combine proximity operators.

:Authors:

* Samuel Farrens <samuel.farrens@cea.fr>,
* Loubna El Gueddari <loubna.elgueddari@gmail.com>
* Pierre-Antoine Comby <pierre-antoine.comby@crans.org>

"""

import numpy as np

from modopt.base.types import check_callable


class ProximityParent(object):
    """Proximity Operator Parent Class.

    This class sets the structure for defining proximity operator instances.

    Parameters
    ----------
    op : callable
        Callable function that implements the proximity operation
    cost : callable
        Callable function that implements the proximity contribution to the
        cost
    """

    def __init__(self, op, cost):

        self.op = op
        self.cost = cost

    @property
    def op(self):
        """Linear operator."""
        return self._op

    @op.setter
    def op(self, operator):

        self._op = check_callable(operator)

    @property
    def cost(self):
        """Cost contribution.

        This method defines the proximity operator's contribution to the total
        cost.

        Returns
        -------
        float
            Cost contribution value

        """
        return self._cost

    @cost.setter
    def cost(self, method):

        self._cost = check_callable(method)


class LinearCompositionProx(ProximityParent):
    """Proximity Operator of a Linear Composition.

    This class defines the proximity operator of a function given by
    a composition between an initial function whose proximity operator is known
    and an orthogonal linear function.

    Parameters
    ----------
    linear_op : class instance
        Linear operator class
    prox_op : class instance
        Proximity operator class

    See Also
    --------
    ProximityParent : parent class

    """

    def __init__(self, linear_op, prox_op):
        self.linear_op = linear_op
        self.prox_op = prox_op
        self.op = self._op_method
        self.cost = self._cost_method

    def _op_method(self, input_data, extra_factor=1.0):
        """Operator.

        This method returns the scaled version of the proximity operator as
        given by Lemma 2.8 of :cite:`combettes2005`.

        Parameters
        ----------
        input_data : numpy.ndarray
            Input data array
        extra_factor : float
            Additional multiplication factor (default is ``1.0``)

        Returns
        -------
        numpy.ndarray
            Result of the scaled proximity operator

        """
        return self.linear_op.adj_op(
            self.prox_op.op(
                self.linear_op.op(input_data),
                extra_factor=extra_factor,
            ),
        )

    def _cost_method(self, *args, **kwargs):
        """Calculate the cost function associated to the composed function.

        Parameters
        ----------
        *args : tuple
            Positional arguments
        **kwargs : dict
            Keyword arguments

        Returns
        -------
        float
            The cost of the associated composed function

        """
        return self.prox_op.cost(self.linear_op.op(args[0]), **kwargs)


class ProximityCombo(ProximityParent):
    """Proximity Combo.

    This class defines a combined proximity operator.

    Parameters
    ----------
    operators : list
        List of proximity operator class instances

    Examples
    --------
    >>> from modopt.opt.proximity import ProximityCombo, ProximityParent
    >>> a = ProximityParent(lambda x: x ** 2, lambda x: x ** 3)
    >>> b = ProximityParent(lambda x: x ** 4, lambda x: x ** 5)
    >>> c = ProximityCombo([a, b])
    >>> c.op([2, 2])
    array([4, 16], dtype=object)
    >>> c.cost([2, 2])
    40

    See Also
    --------
    ProximityParent : parent class

    """

    def __init__(self, operators):

        operators = self._check_operators(operators)
        self.operators = operators
        self.op = self._op_method
        self.cost = self._cost_method

    def _check_operators(self, operators):
        """Check operators.

        This method cheks that the input operators and weights are correctly
        formatted.

        Parameters
        ----------
        operators : list, tuple or numpy.ndarray
            List of linear operator class instances

        Returns
        -------
        numpy.ndarray
            Operators

        Raises
        ------
        TypeError
            For invalid input type
        ValueError
            For empty list
        ValueError
            For missing op method
        ValueError
            For missing cost method

        """
        if not isinstance(operators, (list, tuple, np.ndarray)):
            raise TypeError(
                'Invalid input type, operators must be a list, tuple or '
                + 'numpy array.',
            )

        operators = np.array(operators)

        if not operators.size:
            raise ValueError('Operator list is empty.')

        for operator in operators:
            if not hasattr(operator, 'op'):
                raise ValueError('Operators must contain "op" method.')
            if not hasattr(operator, 'cost'):
                raise ValueError('Operators must contain "cost" method.')
            operator.op = check_callable(operator.op)
            operator.cost = check_callable(operator.cost)

        return operators

    def _op_method(self, input_data, extra_factor=1.0):
        """Operator.

        This method returns the result of applying all of the proximity
        operators to the data.

        Parameters
        ----------
        input_data : numpy.ndarray
            Input data array
        extra_factor : float
            Additional multiplication factor (default is ``1.0``)

        Returns
        -------
        numpy.ndarray
            Result

        """
        res = np.empty(len(self.operators), dtype=np.ndarray)

        for index, _ in enumerate(self.operators):
            res[index] = self.operators[index].op(
                input_data[index],
                extra_factor=extra_factor,
            )

        return res

    def _cost_method(self, *args, **kwargs):
        """Calculate combined proximity operator components of the cost.

        This method returns the sum of the cost components from each of the
        proximity operators.

        Parameters
        ----------
        *args : tuple
            Positional arguments
        **kwargs : dict
            Keyword arguments

        Returns
        -------
        float
            Combinded cost components

        """
        return np.sum([
            operator.cost(input_data)
            for operator, input_data in zip(self.operators, args[0])
        ])
