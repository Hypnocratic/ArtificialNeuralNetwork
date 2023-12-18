import numpy as np

from neurons import Neuron

from typing import Callable

class Network():
    def __init__(self, model:list):
        self.model = model