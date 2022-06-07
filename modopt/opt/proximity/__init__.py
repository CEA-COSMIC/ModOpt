
# -*- coding: utf-8 -*-

"""PROXIMITY OPERATORS.

This module contains classes of proximity operators for optimisation.

:Authors:

* Samuel Farrens <samuel.farrens@cea.fr>,
* Loubna El Gueddari <loubna.elgueddari@gmail.com>
* Pierre-Antoine Comby <pierre-antoine.comby@crans.org>
"""

from modopt.opt.proximity.base import (LinearCompositionProx, ProximityCombo,
                                       ProximityParent)
from modopt.opt.proximity.classic import IdentityProx, Positivity
from modopt.opt.proximity.norms import (ElasticNet, GroupLASSO, KSupportNorm,
                                        OrderedWeightedL1Norm, Ridge,
                                        SparseThreshold)
from modopt.opt.proximity.rank import LowRankMatrix
