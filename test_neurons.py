import unittest
import numpy as np
from random import randint

from neurons import Neuron

def identity_fn(x:float) -> float:
    return x

def identity_fn_prime(x:float) -> int:
    return 1

def squared_error_fn(y_hat:float, y:float, n:int=None) -> float:
    return (y_hat - y)**2

def squared_error_fn_prime(y_hat:float, y:float, n:int=None) -> float:
    return 2*(y_hat - y)

def ground_truth(x:float) -> float:
    return 3*x+1

class TestNeuron(unittest.TestCase):
    def test_compute_linear(self):
        w = 2
        b = 3
        neuron = Neuron(
            w = np.array([w]),
            b = b,
            position = 1,
            activation_fn = identity_fn,
            activation_fn_prime = identity_fn_prime,
            learning_rate = 0.01,
            loss_fn_prime = squared_error_fn_prime,
            last_neuron = True,
            neurons = 1
        )

        for _ in range(10):
            x = np.array([randint(1, 1_000)])
            self.assertEqual(neuron.compute_linear(x=x), x*w+b)
            self.assertEqual(neuron.compute_linear(x=-x), -x*w+b)

    def test_forward(self):
        w = 2
        b = 3
        neuron = Neuron(
            w = np.array([w]),
            b = b,
            position = 1,
            activation_fn = identity_fn,
            activation_fn_prime = identity_fn_prime,
            learning_rate = 0.01,
            loss_fn_prime = squared_error_fn_prime,
            last_neuron = True,
            neurons = 1
        )

        for _ in range(10):
            x = np.array([randint(1, 1_000)])
            self.assertEqual(neuron.forward(x=x), identity_fn(x*w+b))
            self.assertEqual(neuron.forward(x=-x), identity_fn(-x*w+b))
    
    def test_linear_gradient(self):
        w = 2
        b = 3
        neuron = Neuron(
            w = np.array([w]),
            b = b,
            position = 1,
            activation_fn = identity_fn,
            activation_fn_prime = identity_fn_prime,
            learning_rate = 0.01,
            loss_fn_prime = squared_error_fn_prime,
            last_neuron = True,
            neurons = 1
        )

        for _ in range(10):
            x = np.array([randint(1, 1_000)])
            self.assertEqual(neuron.linear_gradient(forward_gradients=None, forward_weights=None,
                                                    x=x, y=ground_truth(x)), 2*(x*w+b - ground_truth(x)))
            self.assertEqual(neuron.linear_gradient(forward_gradients=None, forward_weights=None,
                                                    x=-x, y=ground_truth(-x)), 2*(-x*w+b - ground_truth(-x)))
    
    def test_weights_gradient(self):
        w = 2
        b = 3
        neuron = Neuron(
            w = np.array([w]),
            b = b,
            position = 1,
            activation_fn = identity_fn,
            activation_fn_prime = identity_fn_prime,
            learning_rate = 0.01,
            loss_fn_prime = squared_error_fn_prime,
            last_neuron = True,
            neurons = 1
        )

        for _ in range(10):
            x = np.array([randint(1, 1_000)])
            self.assertEqual(neuron.weights_gradient(z_grad=2*(x*w+b - ground_truth(x)), x=x),
                             2*(x*w+b - ground_truth(x)) * x)
            self.assertEqual(neuron.weights_gradient(z_grad=2*(-x*w+b - ground_truth(-x)), x=-x),
                             2*(-x*w+b - ground_truth(-x)) * -x)

if __name__ == '__main__':
    unittest.main()