import numpy as np

from typing import Callable

class Neuron():
    def __init__(self, w:np.array, b:float, activation_fn:Callable, activation_fn_prime:Callable,
                 learning_rate:float, last_neuron:bool=False, neurons:int=None, loss_fn_prime:Callable=None) -> None:
        '''
        w: numpy array of weights
        b: neuron bias
        activation_fn: Activation Function
        activation_fn_prime: derivative of the activation function
        learning_rate: learning rate
        loss_fn_prime: derivative of loss function
        last_neuron: wheather the neuron is the output neuron
        neurons: number of neurons in same layer (optional if layer is not last)
        '''
        self.w = w
        self.b = b
        self.activation_fn = activation_fn
        self.activation_fn_prime = activation_fn_prime
        self.learning_rate = learning_rate
        self.last_neuron = last_neuron
        self.neurons = neurons
        self.loss_fn_prime = loss_fn_prime
    
    def compute_linear(self, x:np.array) -> float:
        z = np.sum(x * self.w) + self.b
        return z

    def forward(self, x:np.array) -> float: # arguments: inputs
        neuron_output = self.activation_fn(self.compute_linear(x))
        return neuron_output
    
    def linear_gradient(self, forward_gradients:np.array, forward_weights:np.array, x:np.array, y:float=None) -> float:
        z = self.compute_linear(x)
        if self.last_neuron:
            z_grad = 1/self.neurons * self.loss_fn_prime(z, y, self.neurons) * self.activation_fn_prime(z)
        else:
            z_grad = np.sum(forward_gradients * forward_weights) * self.activation_fn_prime(z)
        return z_grad

    def weights_gradient(self, z_grad:float, x:np.array) -> np.array: # arguments: gradients from forward neurons
        w_grad = z_grad * x
        return w_grad

    def bias_gradient(self, z_grad:float) -> float: # arguments: gradients from forward neurons
        b_grad = z_grad
        return b_grad
    
    def update_parameters(self, forward_gradients:np.array, forward_weights:np.array, x:np.array, y:float=None) -> list:
        z_grad = self.linear_gradient(forward_gradients, forward_weights, x, y)
        w_grad = self.weights_gradient(z_grad, x)
        b_grad = self.bias_gradient(z_grad)
        updated_w = self.w - (self.learning_rate * w_grad)
        updated_b = self.b - (self.learning_rate * b_grad)
        return updated_w, updated_b