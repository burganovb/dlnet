"""
Example that can't be learned with 
linear model is xor
"""
import numpy as np
from babanet.train import train
from babanet.nn import NeuralNet
from babanet.layers import Linear, Tanh

inputs = np.array([
    [0,0],
    [1,0],
    [0,1],
    [1,1]
])

targets = np.array([
    [1,0],
    [0,1],
    [0,1],
    [1,0]
])

net = NeuralNet([
    Linear(input_size=2, output_size=2)
])

train(net, inputs, targets)

for x, y in zip(inputs, targets):
    predicted = net.forward(inputs)
    print(x, predicted, y)