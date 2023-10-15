import numpy as np

from typing import Callable

class Neuron():
    def __init__(self, w:np.array, b:float, position:int, activation_fn:Callable, activation_fn_prime:Callable,
                 learning_rate:float, loss_fn_prime:Callable=None, last_neuron:bool=False, neurons:int=None) -> None:
                 # arguments: weights, biases, activation function, activation function derivative,
                 #            loss function derivative, if
        self.w = w
        self.b = b
        self.position = position
        self.activation_fn = activation_fn
        self.activation_fn_prime = activation_fn_prime
        self.learning_rate = learning_rate
        self.loss_fn_prime = loss_fn_prime
        self.last_neuron = last_neuron
        self.neurons = neurons
    
    def forward(self, x:np.array) -> float: # arguments: inputs
        self.x = x
        self.z = np.sum(x * self.w) + self.b
        neuron_output = self.activation_fn(self.z)
        return neuron_output
    
    def backpropagation(self, forward_gradients:np.array, forward_weights:np.array, x:np.array, y:float=None) -> float:
        _ = self.forward(x)
        if self.last_neuron:
            n_grad = 1/self.neurons * self.loss_fn_prime(self.z, y, self.neurons) * self.activation_fn_prime(self.z)
        else:
            n_grad = np.sum(forward_gradients * forward_weights) * self.activation_fn_prime(self.z)
        return n_grad

    def weights_gradient(self, n_grad:float) -> np.array: # arguments: gradients from forward neurons
        w_grad = n_grad * self.x
        return w_grad

    def bias_gradient(self, n_grad:float) -> float: # arguments: gradients from forward neurons
        b_grad = n_grad
        return b_grad
    
    def gradient_descent(self, forward_gradients:np.array, forward_weights:np.array, x:np.array, y:float=None) -> None:
        n_grad = self.backpropagation(forward_gradients, forward_weights, x, y)
        w_grad = self.weights_gradient(n_grad)
        b_grad = self.bias_gradient(n_grad)
        self.w = self.w - (self.learning_rate * w_grad)
        self.b = self.b - (self.learning_rate * b_grad)
        print(f"weight gradient: {w_grad[0]}")
        print(f"bias gradient: {b_grad}")