"""
A loss function measures how good our predictions are

"""

import numpy as np
from babanet.tensor import Tensor

class Loss:
    def loss(self, predicted: Tensor, actual:Tensor) -> float:
        raise NotImplementedError

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError


class MSE(Loss):
    """
    MSE is mean squared error, but this is total squred error
    """
    def loss(self, predicted: Tensor, actual:Tensor) -> float:
        return np.sum((predicted-actual)**2)

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return 2*(predicted-actual)
