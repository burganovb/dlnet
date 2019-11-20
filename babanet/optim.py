"""
Uses an optimizer to adjust the parameters
of the network based on the gradients computed 
during backprop
"""

from babanet.nn import NeuralNet

class Optimizer:
    def step(self, net: NeuralNet) -> None:
        raise NotImplementedError

