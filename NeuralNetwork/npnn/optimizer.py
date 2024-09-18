"""18-661 HW5 Optimization Policies."""

import numpy as np

from .base import Optimizer


class SGD(Optimizer):
    """Simple SGD optimizer.

    Parameters
    ----------
    learning_rate : float
        SGD learning rate.
    """

    def __init__(self, learning_rate=0.01):

        self.learning_rate = learning_rate

    def apply_gradients(self, params):
        """Apply gradients to parameters.

        Parameters
        ----------
        params : Variable[]
            List of parameters that the gradients correspond to.
        """
        for p in params:
            p.value -= self.learning_rate * p.grad

class Adam(Optimizer):
    """Adam (Adaptive Moment) optimizer.

    Parameters
    ----------
    learning_rate : float
        Learning rate multiplier.
    beta1 : float
        Momentum decay parameter.
    beta2 : float
        Variance decay parameter.
    epsilon : float
        A small constant added to the demoniator for numerical stability.
    """

    def __init__(
            self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-7, time_step=0):

        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.time_step = time_step

    def initialize(self, params):
        """Initialize any optimizer state needed.

        params : np.array[]
            List of parameters that will be used with this optimizer.
        """
        self.mt = []
        self.vt = []
        for p in params:
            self.mt.append(np.zeros(p.value.shape))
            self.vt.append(np.zeros(p.value.shape))
        

    def apply_gradients(self, params):
        """Apply gradients to parameters.

        Parameters
        ----------
        params : Variable[]
            List of parameters that the gradients correspond to.
        """
        self.time_step += 1
        for p in range(len(params)):
            
            self.mt[p] = self.beta1 * self.mt[p] + (1-self.beta1)*(params[p].grad)
            self.vt[p] = self.beta2 * self.vt[p] + (1 -self.beta2)* (params[p].grad)**2
            
            m_hat = self.mt[p]/(1-self.beta1**self.time_step)
            v_hat = self.vt[p]/(1-self.beta2**self.time_step)
            
            params[p].value -= (self.learning_rate / ((np.sqrt(v_hat) + self.epsilon))) * m_hat
