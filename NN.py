import numpy as np
from random import randint

from Neurons import Neuron

def identity_fn(x:float) -> float:
    return x

def identity_fn_prime(x:float) -> int:
    return 1

def squared_error_fn(y_hat:float, y:float, n:int=None) -> float:
    return (y_hat - y)**2

def squared_error_fn_prime(y_hat:float, y:float, n:int=None) -> float:
    return 2*(y_hat - y)

def fn(x:float) -> float:
    return x

init_neuron = Neuron(
        w = np.array([1]),
        b = 0,
        position = 1,
        activation_fn = identity_fn,
        activation_fn_prime = identity_fn_prime,
        learning_rate = 0.001,
        loss_fn_prime = squared_error_fn_prime,
        last_neuron = True,
        neurons = 1
    )

def train(neuron:Neuron, epochs:int):
    for epoch in range(epochs):
        if epoch % 1 == 0:
            print(f"epoch: {epoch}, weight: {neuron.w}, bias: {neuron.b}")
            pass
        x = randint(1, 1_000)
        neuron.gradient_descent(
            forward_gradients = None,
            forward_weights = None,
            x = np.array([x]),
            y = fn(x)
        )
    return neuron

trained_neuron = train(init_neuron, epochs=100)

for n in [1, 2, 5, 7.5, 100]:
    pred = trained_neuron.forward(np.array([n]))
    print(f"x = {n}, y = {fn(n)}, prediction = {pred}, Loss = {squared_error_fn(pred, fn(n))}")

print(f"weight: {trained_neuron.w}")
print(f"bias: {trained_neuron.b}")