"""Gradient Descent Algorithms."""
import numpy as np

# Third party import
from modopt.opt.algorithms import SetUp
from modopt.opt.cost import costObj


class GenericGradOpt(SetUp):
    r"""Generic Gradient descent operator.

    Performs the descent algorithm in the direction m_k at speed s_k.

    ..math:: x_{k+1} = prox(x_k - \frac{\eta}{\sqrt{s_k + \epsilon}} m_k)
    """

    def __init__(self, x, grad, prox, cost,
                 eta=1.0, eta_update=None, epsilon=1e-6, epoch_size=1,
                 metric_call_period=5, metrics=None, **kwargs):
        """Create generic gradient descent algorithm.

        Parameters
        ----------
        x: ndarray
            Initial value
        grad: Instance of GradBase
            Gradient operator
        prox: Instance of ProximalOperator
            Proximal operator,
        linear: Instance of OperatorBase
            Linear operator (the image domain should be sparse)
        cost:
            Cost Operator
        eta: float, default 1.0
            Descent step
        eta_update: callable, default None
            If not None, used to update eta at each step.
        epsilon: float, default 1e-6
            Numerical stability constant for the gradient.
        epoch_size, int, default 1
        metric_call_period: int, default 5
            The period of iteration on which metrics will be computed.
        metrics: dict, default None
            If not None, specify which metrics to use.

        See Also
        --------
        metric api
        Setup
        """
        # Set the initial variable values
        if metrics is None:
            metrics = dict()
        # Set default algorithm properties
        super().__init__(
            metric_call_period=metric_call_period, metrics=metrics, **kwargs
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
        for param in (eta, epsilon):
            self._check_param(param)
        self._eta = eta
        self._eps = epsilon

        # Set the algorithm parameter update methods
        self._check_param_update(eta_update)
        self._eta_update = eta_update
        self.idx = 0
        self.epoch_size = epoch_size

    def _update(self):
        self._grad.get_grad(self._x_old)
        self.update_grad_dir(self._grad.grad)
        self.update_grad_speed(self._grad.grad)
        step = self._eta / (np.sqrt(self._speed_grad) + self._eps)
        self._x_new = self._x_old - step * self._dir_grad
        if self.idx % self.epoch_size == 0:
            self.reset()
            self.update_reg(step)
        self._x_old = self._x_new.copy()
        if self._eta_update is not None:
            self._eta = self._eta_update(self._eta, self.idx)
        # Test cost function for convergence.
        if self._cost_func:
            self.converge = (self.any_convergence_flag()
                             or self._cost_func.get_cost(self._x_new))

    def update_grad_dir(self, grad):
        """Update the gradient descent direction."""
        self._dir_grad = grad

    def update_grad_speed(self, grad):
        """Update the gradient descent speed."""
        pass

    def update_reg(self, factor):
        """Regularisation step."""
        self._x_new = self._prox.op(self._x_new, extra_factor=factor)

    def get_notify_observers_kwargs(self):
        """Notify observers.

        Return the mapping between the metrics call and the iterated
        variables.

        Returns
        -------
        notify_observers_kwargs : dict,
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
    """

    def __init__(self, *args, **kwargs):
        """Create Vanilla Descent Algorithm.

        See Also
        --------
        GenericGradOpt
        """
        super().__init__(*args, **kwargs)
        # no scale factor
        self._speed_grad = 1.0
        self._eps = 0


class AdaGenericGradOpt(GenericGradOpt):
    """Generic Grad descent Algorithm with ADA acceleration scheme."""

    def update_grad_speed(self, grad):
        """Ada Acceleration Scheme."""
        self._speed_grad += abs(grad) ** 2


class RMSpropGradOpt(GenericGradOpt):
    r"""RMSprop Gradient descent algorithm.

    The gradient speed is defined as:
    s_k = \gamma s_{k-1}  + (1-\gamma) * |\nabla f|^2
    """

    def __init__(self, *args, gamma=0.5, **kwargs):
        """Create RMSprop algorithm.

        Parameters
        ---------
        gamma: float, default 0.5

        Raises
        ------
        ValueError
            If gamma is outside ]0,1[
        """
        super().__init__(*args, **kwargs)
        if gamma < 0 or gamma > 1:
            raise ValueError("gamma is outside of range [0,1]")
        self._check_param(gamma)
        self._gamma = gamma

    def update_grad_speed(self, grad):
        """Rmsprop update speed."""
        self._speed_grad = (
            self._gamma * self._speed_grad + (1 - self._gamma) * abs(grad) ** 2
        )


class MomentumGradOpt(GenericGradOpt):
    r"""Momentum (Heavy-ball) descent algorithm.

    ..math:: m_k = \beta * m_{k-1} + \nabla f(x_k)
    """

    def __init__(self, *args, beta=0.9, **kwargs):
        """Create Momentuk Algorithm.

        Parameters
        ----------
        beta: float, default 0.9
        """
        super().__init__(*args, **kwargs)
        self._check_param(beta)
        self._beta = beta
        # no scale factor
        self._speed_grad = 1.0
        self._eps = 0.0

    def update_grad_dir(self, grad):
        """Momentum gradient direction update."""
        self._dir_grad = self._beta * self._dir_grad + grad

    def reset(self):
        """Reset the gradient direction."""
        self._dir_grad = np.zeros_like(self._x_new)


class ADAMGradOpt(GenericGradOpt):
    r"""ADAM optimizer.

    ..math:: m_{k+1} = \frac{1}{1-\beta^k}(\beta*m_{k}+(1-\beta)*|\nabla f_k|^2)
    ..math:: s_{k+1} = \frac{1}{1-\gamma^k}(\gamma*s_k+(1-\gamma)*\nabla f_k)

    """

    def __init__(self, *args, **kwargs):
        """
        Create ADAM Optimiser.

        Parameters
        ----------
        gamma: float
        beta: float

        Raises
        -----
        ValueError
            If gamma or beta is outside ]0,1[
        """
        gamma = kwargs.pop("gamma")
        beta = kwargs.pop("beta")
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

    def update_grad_dir(self, grad):
        """ADAM Update of gradient direction."""
        self._beta_pow *= self._beta

        self._dir_grad = (1.0 / (1.0 - self._beta_pow)) * (
            self._beta * self._dir_grad + (1 - self._beta) * grad
        )

    def update_grad_speed(self, grad):
        """ADAM Updatae of gradient speed."""
        self._gamma_pow *= self._gamma
        self._speed_grad = (1.0 / (1.0 - self._gamma_pow)) * (
            self._gamma * self._speed_grad + (1 - self._gamma) * abs(grad) ** 2
        )


class SAGAOptGradOpt(GenericGradOpt):
    """SAGA optimizer.

    Notes
    -----
    The stochastic part is not handled here, and should be implemented by
    changing the obs_data between each call to the _update function.
    """

    def __init__(self, *args, **kwargs):
        """Create SAGA Optimizer."""
        super().__init__(*args, **kwargs)
        self._grad_memory = np.zeros(
            (self.epoch_size, *self._x_old.size), dtype=self._x_old.dtype
        )

    def update_grad_dir(self, grad):
        """SAGA Update gradient direction."""
        cycle = self.idx % self.epoch_size
        self._dir_grad = self._dir_grad - self._grad_memory[cycle] + grad
        self._grad_memory[cycle] = grad
