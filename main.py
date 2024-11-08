import numpy as np
from activation_functions import sigmoid, dsigmoid


class Neuron:

    def __init__(self, num_inputs):
        self._num_inputs = num_inputs
        self._w = np.random.rand(num_inputs)
        self._last_input = None
        self._last_output = None

    def query(self, x):
        self._x = np.array(x)
        self._last_output = sigmoid(np.dot(self._w, self._x))
        self._last_input = self._x

    def train(self, target, learning_rate=0.01):
        error = target - self._last_output  # absolute error
        self._w = self._w - learning_rate * (error * dsigmoid(self._last_output) * self._last_input)
        return self._w
