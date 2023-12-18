import numpy as np
from neurons import Neuron

from typing import Callable

class DenseLayer():
    def __init__(self, neurons:int, w:np.array, b:np.array, activation_fn:Callable, activation_fn_prime:Callable,
                 learning_rate:float, last_layer:bool = False, loss_fn_prime:Callable = None) -> None:
        self.neurons = neurons
        self.w = w
        self.b = b
        self.activation_fn = activation_fn
        self.activation_fn_prime = activation_fn_prime
        self.learning_rate = learning_rate
        self.loss_fn_prime = loss_fn_prime
        self.last_layer = last_layer

        layer = []
        if last_layer:
            for i in range(neurons):
                neuron = Neuron(
                    w=w[i],
                    b=b[i],
                    activation_fn=activation_fn,
                    activation_fn_prime=activation_fn_prime,
                    learning_rate=learning_rate,
                    last_neuron=True,
                    neurons=neurons,
                    loss_fn_prime=loss_fn_prime
                )
                layer.append(neuron)
        else:
            for i in range(neurons):
                neuron = Neuron(
                    w=w[i],
                    b=b[i],
                    activation_fn=activation_fn,
                    activation_fn_prime=activation_fn_prime,
                    learning_rate=learning_rate
                )
                layer.append(neuron)
        self.layer = layer
    
    def compute_linear(self, x:np.array) -> float:
        return super().compute_linear(x)

    # x = [[],[],[]]
    def forward(self, x:np.array) -> np.array:
        z = 

    def linear_gradient(self, forward_gradients:np.array, forward_weights:np.array, x:np.array, y:float = None) -> float:
        return super().linear_gradient(forward_gradients, forward_weights, x, y)
    
    def weights_gradient(self, z_grad:float, x:np.array) -> np.array:
        return super().weights_gradient(z_grad, x)
    
    def bias_gradient(self, z_grad:float) -> float:
        return super().bias_gradient(z_grad)
    
    def update_parameters(self, forward_gradients:np.array, forward_weights:np.array, x:np.array, y:float = None) -> list:
        return super().update_parameters(forward_gradients, forward_weights, x, y)

def relu(x):
    return max(0, x)

def relu_prime(x):
    if x <= 0:
        return 0
    else:
        return 1

weights = np.random.rand(5, 3)
bias = np.random.rand(5)

layer = DenseLayer(
    neurons=5,
    w=weights,
    b=bias,
    activation_fn=relu,
    activation_fn_prime=relu_prime,
    learning_rate=10**-6
)

print(layer.construct_layer()[0].b)