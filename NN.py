import numpy as np
from random import randint

from neurons import Neuron

def identity_fn(x:float) -> float:
    return x

def identity_fn_prime(x:float) -> int:
    return 1

def relu(x):
    return max(0, x)

def relu_prime(x):
    if x <= 0:
        return 0
    else:
        return 1

def squared_error_fn(y_hat:float, y:float, n:int=None) -> float:
    return (y_hat - y)**2

def squared_error_fn_prime(y_hat:float, y:float, n:int=None) -> float:
    return 2*(y_hat - y)

def fn(x:float) -> float:
    return 3*x + 10

init_neuron = Neuron(
        w = np.array([1]),
        b = 1,
        position = 1,
        activation_fn = identity_fn,
        activation_fn_prime = identity_fn_prime,
        learning_rate = 10**-6,
        loss_fn_prime = squared_error_fn_prime,
        last_neuron = True,
        neurons = 1
    )

inpt = [709, 942, 603, 508, 420, 846, 835, -98, -900, 243, -106, -878, -557, -460, 197, -382, 758, 966, -221, -913, -98, 145, 990, 841,
        707, -829, -419, -739, 534, 256, -858, -275, 126, 680, 540, -917, -339, 276, -173, 698, -822, 365, -880, -638, 882, 34, 80, 626,
        121, 623, -515, 956, 497, 986, 471, -785, -491, 681, 209, 54, -465, 347, -742, -794, 475, -24, -414, 880, -896, 184, 724, -447,
        558, 733, 56, 528, -942, 221, -827, -318, -975, -440, 415, 273, 794, 343, 319, -201, -683, -790, 187, -39, -317, -551, 229, 711,
        -285, 287, 624, 53]

def single_epoch_train(neuron:Neuron, dataset:list):
    for step in range(len(dataset)):
        x = dataset[step]
        updated_parameters = neuron.update_parameters(
            forward_gradients = None,
            forward_weights = None,
            x = np.array([x]),
            y = fn(x)
        )
        neuron.w = updated_parameters[0]
        neuron.b = updated_parameters[1]
    return neuron

def train(neuron:Neuron, dataset:list, epochs:int):
    if epochs == 1:
        trained_neuron = single_epoch_train(neuron, dataset)
        return trained_neuron
    else:
        return train(single_epoch_train(neuron, dataset), dataset, epochs-1)

def train(neuron:Neuron, dataset:list, epochs:int):
    for epoch in range(epochs):
        neuron = single_epoch_train(neuron, dataset)
    return neuron

trained_neuron = train(
    neuron=init_neuron,
    dataset=inpt,
    epochs=50_000
)

print(trained_neuron)

for n in [1, 2, 5, 7.5, 100]:
    pred = trained_neuron.forward(np.array([n]))
    print(f"x = {n}, y = {fn(n)}, prediction = {pred}, Loss = {squared_error_fn(pred, fn(n))}")

print(f"weight: {trained_neuron.w}")
print(f"bias: {trained_neuron.b}")