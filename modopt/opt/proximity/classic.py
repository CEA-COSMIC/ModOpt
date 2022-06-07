"""Classical Proximal operators."""

import numpy as np
from modopt.opt.proximity.base import ProximityParent
from modopt.signal.positivity import positive


class IdentityProx(ProximityParent):
    """Identity Proxmity Operator.

    This is a dummy class that can be used as a proximity operator.

    Notes
    -----
    The identity proximity operator contributes ``0.0`` to the total cost.

    See Also
    --------
    ProximityParent : parent class

    """

    def __init__(self):

        self.op = lambda x_val: x_val
        self.cost = lambda x_val: 0


class Positivity(ProximityParent):
    """Positivity Proximity Operator.

    This class defines the positivity proximity operator.

    See Also
    --------
    ProximityParent : parent class
    modopt.signal.positivity.positive : positivity operator

    """

    def __init__(self):

        self.op = positive
        self.cost = self._cost_method

    def _cost_method(self, *args, **kwargs):
        """Calculate positivity component of the cost.

        This method returns ``0`` as the posivituty does not contribute to the
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
            ``0.0``

        """
        if 'verbose' in kwargs and kwargs['verbose']:
            print(' - Min (X):', np.min(args[0]))

        return 0
