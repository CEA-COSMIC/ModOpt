"""Gradient Descent Algorithms."""

import numpy as np

from modopt.opt.algorithms.base import SetUp
from modopt.opt.cost import costObj


class GenericGradOpt(SetUp):
    r"""Generic Gradient descent operator.

    Performs the descent algorithm in the direction :math:`m_k` at speed
    :math:`s_k`.


    Parameters
    ----------
    x: numpy.ndarray
        Initial value
    grad
        Gradient operator class instance
    prox
        Proximity operator class instance
    cost : class instance or str, optional
        Cost function class instance (default is ``'auto'``); Use ``'auto'`` to
        automatically generate a ``costObj`` instance
    eta: float
        Descent step, :math:`\eta` (default is ``1.0``)
    eta_update: callable
        If not ``None``, used to update :math:`\eta` at each step
        (default is ``None``)
    epsilon: float
        Numerical stability constant for the gradient, :math:`\epsilon`
        (default is ``1e-6``)
    epoch_size: int
        Size of epoch for the descent (default is ``1``)
    metric_call_period: int
        The period of iteration on which metrics will be computed
        (default is ``5``)
    metrics: dict
        If not None, specify which metrics to use (default is ``None``)

    Notes
    -----
    The Gradient descent  step is defined as:

    .. math:: x_{k+1} = x_k - \frac{\eta}{\sqrt{s_k + \epsilon}} m_k

    where:

    * :math:`m_k` is the gradient direction
    * :math:`\eta` is the gradient descent step
    * :math:`s_k` is the gradient "speed"

    At each Epoch, an optional Proximal step can be performed.

    The following state variable are available for metrics measurememts:

    * ``'x_new'`` : new estimate of the iterations
    * ``'dir_grad'`` : direction of the gradient descent step
    * ``'speed_grad'`` : speed for the gradient descent step
    * ``'idx'`` : index of the iteration being reconstructed.

    See Also
    --------
    modopt.opt.algorithms.base.SetUp : parent class
    modopt.opt.cost.costObj : cost object class

    """

    def __init__(
        self,
        x,
        grad,
        prox,
        cost,
        eta=1.0,
        eta_update=None,
        epsilon=1e-6,
        epoch_size=1,
        metric_call_period=5,
        metrics=None,
        **kwargs,
    ):
        # Set the initial variable values
        if metrics is None:
            metrics = {}
        # Set default algorithm properties
        super().__init__(
            metric_call_period=metric_call_period,
            metrics=metrics,
            **kwargs,
        )
        self.iter = 0
        self._check_input_data(x)
        self._x_old = np.copy(x)
        self._x_new = np.copy(x)
        self._speed_grad = np.zeros(x.shape, dtype=float)
        self._dir_grad = np.zeros_like(x)
        # Set the algorithm operators
        for operator in (grad, prox, cost):
            self._check_operator(operator)
        self._grad = grad
        self._prox = prox
        if cost == "auto":
            self._cost_func = costObj([self._grad, self._prox])
        else:
            self._cost_func = cost
        # Set the algorithm parameters
        for param_val in (eta, epsilon):
            self._check_param(param_val)
        self._eta = eta
        self._eps = epsilon

        # Set the algorithm parameter update methods
        self._check_param_update(eta_update)
        self._eta_update = eta_update
        self.idx = 0
        self.epoch_size = epoch_size

    def iterate(self, max_iter=150):
        """Iterate.

        This method calls update until either convergence criteria is met or
        the maximum number of iterations is reached.

        Parameters
        ----------
        max_iter : int, optional
            Maximum number of iterations (default is ``150``)

        """
        self._run_alg(max_iter)

        # retrieve metrics results
        self.retrieve_outputs()

        self.x_final = self._x_new

    def _update(self):
        """Update.

        This method updates the current reconstruction.

        """
        self._grad.get_grad(self._x_old)
        self._update_grad_dir(self._grad.grad)
        self._update_grad_speed(self._grad.grad)
        step = self._eta / (np.sqrt(self._speed_grad) + self._eps)
        self._x_new = self._x_old - step * self._dir_grad
        if self.idx % self.epoch_size == 0:
            self.reset()
            self._update_reg(step)
        self._x_old = self._x_new.copy()
        if self._eta_update is not None:
            self._eta = self._eta_update(self._eta, self.idx)
        # Test cost function for convergence.
        if self._cost_func:
            self.converge = self.any_convergence_flag() or self._cost_func.get_cost(
                self._x_new
            )

    def _update_grad_dir(self, grad):
        """Update the gradient descent direction.

        Parameters
        ----------
        grad: numpy.ndarray
            The gradient direction

        """
        self._dir_grad = grad

    def _update_grad_speed(self, grad):
        """Update the gradient descent speed.

        Parameters
        ----------
        grad: numpy.ndarray
            The gradient direction

        """
        pass

    def _update_reg(self, factor):
        """Regularisation step.

        Parameters
        ----------
        factor: float or numpy.ndarray
            Extra factor for the proximal step

        """
        self._x_new = self._prox.op(self._x_new, extra_factor=factor)

    def get_notify_observers_kwargs(self):
        """Notify observers.

        Return the mapping between the metrics call and the iterated
        variables.

        Returns
        -------
        dict
           The mapping between the iterated variables

        """
        return {
            "x_new": self._x_new,
            "dir_grad": self._dir_grad,
            "speed_grad": self._speed_grad,
            "idx": self.idx,
        }

    def retrieve_outputs(self):
        """Retrieve outputs.

        Declare the outputs of the algorithms as attributes: x_final,
        y_final, metrics.

        """
        metrics = {}
        for obs in self._observers["cv_metrics"]:
            metrics[obs.name] = obs.retrieve_metrics()
        self.metrics = metrics

    def reset(self):
        """Reset internal state of the algorithm."""
        pass


class VanillaGenericGradOpt(GenericGradOpt):
    """Vanilla Descent Algorithm.

    Fixed step size and no numerical precision threshold.

    See Also
    --------
    GenericGradOpt : parent class

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # no scale factor
        self._speed_grad = 1.0
        self._eps = 0


class AdaGenericGradOpt(GenericGradOpt):
    r"""Generic Grad descent Algorithm with ADA acceleration scheme.

    Notes
    -----
    For AdaGrad (Section 4.2 of :cite:`ruder2017`) the gradient is
    preconditioned using a speed update:

    .. math:: s_k = \sum_{i=0}^k g_k * g_k


    See Also
    --------
    GenericGradOpt : parent class

    """

    def _update_grad_speed(self, grad):
        """Ada Acceleration Scheme.

        Parameters
        ----------
        grad: numpy.ndarray
            The new gradient for updating the speed

        """
        self._speed_grad += abs(grad) ** 2


class RMSpropGradOpt(GenericGradOpt):
    r"""RMSprop Gradient descent algorithm.

    Parameters
    ----------
    gamma: float
        Update weight for the speed of descent, :math:`\gamma`
        (default is ``0.5``)

    Raises
    ------
    ValueError
        If :math:`\gamma` is outside :math:`]0,1[`

    Notes
    -----
    The gradient speed of RMSProp (Section 4.5 of :cite:`ruder2017`) is
    defined as:

    .. math:: s_k = \gamma s_{k-1}  + (1-\gamma) * |\nabla f|^2

    See Also
    --------
    GenericGradOpt : parent class

    """

    def __init__(self, *args, gamma=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        if gamma < 0 or gamma > 1:
            raise ValueError("gamma is outside of range [0,1]")
        self._check_param(gamma)
        self._gamma = gamma

    def _update_grad_speed(self, grad):
        """Rmsprop update speed.

        Parameters
        ----------
        grad: numpy.ndarray
            The new gradient for updating the speed

        """
        self._speed_grad = (
            self._gamma * self._speed_grad + (1 - self._gamma) * abs(grad) ** 2
        )


class MomentumGradOpt(GenericGradOpt):
    r"""Momentum (Heavy-ball) descent algorithm.

    Parameters
    ----------
    beta: float
        update weight for the momentum, :math:`\beta` (default is ``0.9``)

    Notes
    -----
    The Momentum (Section 4.1 of :cite:`ruder2017`) update is defined as:

    .. math:: m_k = \beta * m_{k-1} + \nabla f(x_k)

    See Also
    --------
    GenericGradOpt : parent class

    """

    def __init__(self, *args, beta=0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self._check_param(beta)
        self._beta = beta
        # no scale factor
        self._speed_grad = 1.0
        self._eps = 0

    def _update_grad_dir(self, grad):
        """Momentum gradient direction update.

        Parameters
        ----------
        grad: numpy.ndarray
            The new gradient for updating the speed

        """
        self._dir_grad = self._beta * self._dir_grad + grad

    def reset(self):
        """Reset the gradient direction."""
        self._dir_grad = np.zeros_like(self._x_new)


class ADAMGradOpt(GenericGradOpt):
    r"""ADAM optimizer.

    Parameters
    ----------
    gamma: float
        Update weight, :math:`\gamma`, for the direction in :math:`]0,1[`
    beta: float
        Update weight, :math:`\beta`, for the speed in :math:`]0,1[`

    Raises
    ------
    ValueError
        If gamma or beta is outside :math:`]0,1[`

    Notes
    -----
    The ADAM optimizer (Section 4.6 of :cite:`ruder2017`) is defined as:

    .. math::
        m_{k+1} = \frac{1}{1-\beta^k}(\beta*m_{k}+(1-\beta)*|\nabla f_k|^2)
    .. math::
        s_{k+1} = \frac{1}{1-\gamma^k}(\gamma*s_k+(1-\gamma)*\nabla f_k)

    See Also
    --------
    GenericGradOpt : parent class

    """

    def __init__(self, *args, gamma=0.9, beta=0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self._check_param(gamma)
        self._check_param(beta)
        if gamma < 0 or gamma >= 1:
            raise ValueError("gamma is outside of range [0,1]")
        if beta < 0 or beta >= 1:
            raise ValueError("beta is outside of range [0,1]")
        self._gamma = gamma
        self._beta = beta
        self._beta_pow = 1
        self._gamma_pow = 1

    def _update_grad_dir(self, grad):
        """ADAM Update of gradient direction."""
        self._beta_pow *= self._beta

        self._dir_grad = (1.0 / (1.0 - self._beta_pow)) * (
            self._beta * self._dir_grad + (1 - self._beta) * grad
        )

    def _update_grad_speed(self, grad):
        """ADAM Updatae of gradient speed."""
        self._gamma_pow *= self._gamma
        self._speed_grad = (1.0 / (1.0 - self._gamma_pow)) * (
            self._gamma * self._speed_grad + (1 - self._gamma) * abs(grad) ** 2
        )


class SAGAOptGradOpt(GenericGradOpt):
    """SAGA optimizer.

    Implements equation 7 of :cite:`defazio2014`.

    Notes
    -----
    The stochastic part is not handled here, and should be implemented by
    changing the ``obs_data`` between each call to the ``_update`` function.

    See Also
    --------
    GenericGradOpt : parent class

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._grad_memory = np.zeros(
            (self.epoch_size, *self._x_old.shape),
            dtype=self._x_old.dtype,
        )

    def _update_grad_dir(self, grad):
        """SAGA Update gradient direction.

        Parameters
        ----------
        grad: numpy.ndarray
            The new gradient for updating the speed

        """
        cycle = self.idx % self.epoch_size
        self._dir_grad = self._dir_grad - self._grad_memory[cycle] + grad
        self._grad_memory[cycle] = grad
