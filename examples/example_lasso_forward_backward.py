"""
Solving the LASSO Problem with the Forward Backward Algorithm.
==============================================================

This an example to show how to solve an example LASSO Problem
using the Forward-Backward Algorithm.

In this example we are going to use:
 - Modopt Operators (Linear, Gradient, Proximal)
 - Modopt implementation of solvers
 - Modopt Metric API.
TODO: add reference to LASSO paper.
"""

import numpy as np
import matplotlib.pyplot as plt

from modopt.opt.algorithms import ForwardBackward, POGM
from modopt.opt.cost import costObj
from modopt.opt.linear import LinearParent, Identity
from modopt.opt.gradient import GradBasic
from modopt.opt.proximity import SparseThreshold
from modopt.math.matrix import PowerMethod
from modopt.math.stats import mse

# %%
# Here we create a instance of the LASSO Problem

BETA_TRUE = np.array(
    [3.0, 1.5, 0, 0, 2, 0, 0, 0]
)  # 8 original values from lLASSO Paper
DIM = len(BETA_TRUE)


rng = np.random.default_rng()
sigma_noise = 1
obs = 20
# create a measurement matrix with decaying covariance matrix.
cov = 0.4 ** abs((np.arange(DIM) * np.ones((DIM, DIM))).T - np.arange(DIM))
x = rng.multivariate_normal(np.zeros(DIM), cov, obs)

y = x @ BETA_TRUE
y_noise = y + (sigma_noise * np.random.standard_normal(obs))


# %%
# Next we create Operators for solving the problem.

# MatrixOperator could also work here.
lin_op = LinearParent(lambda b: x @ b, lambda bb: x.T @ bb)
grad_op = GradBasic(y_noise, op=lin_op.op, trans_op=lin_op.adj_op)

prox_op = SparseThreshold(Identity(), 1, thresh_type="soft")

# %%
# In order to get the best convergence rate, we first determine the Lipschitz constant of the gradient Operator
#

calc_lips = PowerMethod(grad_op.trans_op_op, 8, data_type="float32", auto_run=True)
lip = calc_lips.spec_rad
print("lipschitz constant:", lip)

# %%
# Solving using FISTA algorithm
# -----------------------------
#
# TODO: Add description/Reference of FISTA.

cost_op_fista = costObj([grad_op, prox_op], verbose=False)

fb_fista = ForwardBackward(
    np.zeros(8),
    beta_param=1 / lip,
    grad=grad_op,
    prox=prox_op,
    cost=cost_op_fista,
    metric_call_period=1,
    auto_iterate=False,  # Just to give us the pleasure of doing things by ourself.
)

fb_fista.iterate()

# %%
# After the run we can have a look at the results

print(fb_fista.x_final)
mse_fista = mse(fb_fista.x_final, BETA_TRUE)
plt.stem(fb_fista.x_final, label="estimation", linefmt="C0-")
plt.stem(BETA_TRUE, label="reference", linefmt="C1-")
plt.legend()
plt.title(f"FISTA Estimation MSE={mse_fista:.4f}")

# sphinx_gallery_start_ignore
assert mse(fb_fista.x_final, BETA_TRUE) < 1
# sphinx_gallery_end_ignore


# %%
# Solving Using the POGM Algorithm
# --------------------------------
#
# TODO: Add description/Reference to POGM.


cost_op_pogm = costObj([grad_op, prox_op], verbose=False)

fb_pogm = POGM(
    np.zeros(8),
    np.zeros(8),
    np.zeros(8),
    np.zeros(8),
    beta_param=1 / lip,
    grad=grad_op,
    prox=prox_op,
    cost=cost_op_pogm,
    metric_call_period=1,
    auto_iterate=False,  # Just to give us the pleasure of doing things by ourself.
)

fb_pogm.iterate()

# %%
# After the run we can have a look at the results

print(fb_pogm.x_final)
mse_pogm = mse(fb_pogm.x_final, BETA_TRUE)

plt.stem(fb_pogm.x_final, label="estimation", linefmt="C0-")
plt.stem(BETA_TRUE, label="reference", linefmt="C1-")
plt.legend()
plt.title(f"FISTA Estimation MSE={mse_pogm:.4f}")
#
# sphinx_gallery_start_ignore
assert mse(fb_pogm.x_final, BETA_TRUE) < 1
# sphinx_gallery_end_ignore

# %%
# Comparing the Two algorithms
# ----------------------------

plt.figure()
plt.semilogy(cost_op_fista._cost_list, label="FISTA convergence")
plt.semilogy(cost_op_pogm._cost_list, label="POGM convergence")
plt.xlabel("iterations")
plt.ylabel("Cost Function")
plt.legend()
plt.show()


# %%
# We can see that the two algorithm converges quickly, and POGM requires less iterations.
# However the POGM iterations are more costly, so a proper benchmark with time measurement is needed.
# Check the benchopt benchmark for more details.
