""""
The neural net is made up of layers. 
Each layer need sto pass its imputs forward
and propagate gradients backwards. For example,
a neural net might look like

inputs -> Linear -> Tanh -> Linear -> output
"""
from typing import Dict, Callable
import numpy as np
from babanet.tensor import Tensor


class Layer:
    def __init__(self) -> None:
        self.params: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Produce outputs corresponding to these inputs
        """
        raise NotImplementedError

    def backward(self, grad:Tensor) -> Tensor:
        """
        Backpropagate this gradient through the layer
        """
        raise NotImplementedError

class Linear(Layer):
    """
    computes output = inputs @ w + b
    """
    def __init__(self, input_size: int, output_size: int) -> None:
        # inputs are (batch_size, INput_size)
        # outputs are (batch_size, output_size)
        super().__init__()
        self.params["w"] = np.random.randn(input_size, output_size)
        self.params["b"] = np.random.randn(output_size)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        outputs = inputs @ w + b
        """
        self.inputs = inputs
        return inputs @ self.params["w"] + self.params["b"]

    def backward(self, grad: Tensor) -> Tensor:
        """
        backprops gradient
        """
        self.grads["b"] = np.sum(grad, axis=0)
        self.grads["w"] = self.inputs.T @ grad
        return grad @ self.params["w"].T

F = Callable[[Tensor], Tensor]

class Activation(Layer):
    """
    applies a func elementwise to its inputs
    """
    def __init__(self, f:F, f_prime:F) -> None:
        super().__init__()
        self.f = f
        self.f_prime = f_prime
    
    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return self.f(inputs)

    def backward(self, grad:Tensor) -> Tensor:
        return self.f_prime(self.inputs) * grad

def tanh(x:Tensor) -> Tensor:
    return np.tanh(x)

def tanh_prime(x:Tensor) -> Tensor:
    y = tanh(x)
    return 1 - y ** 2

class Tanh(Activation):
    def __init__(self):
        super().__init__(tanh, tanh_prime)