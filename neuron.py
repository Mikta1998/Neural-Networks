import numpy as np
from activation_functions import sigmoid, dsigmoid


class Neuron:

    def __init__(self, num_inputs):
        self._num_inputs = num_inputs
        self._w = np.random.rand(num_inputs)
        self._last_input = None
        self._last_output = None
        self._error = None

    def query(self, x):
        self._x = np.array(x)
        self._last_output = sigmoid(np.dot(self._w, self._x))
        self._last_input = self._x
        return self._last_output

    def train(self, target, learning_rate=0.01):
        self._error = target - self._last_output
        if self._last_output > target:
            dE_do = -1
        else:
            dE_do = 1
        
        self._w = self._w + learning_rate * \
            (dE_do * dsigmoid(self._last_output) * self._last_input)
        return self._w


my_neuron = Neuron(num_inputs=2)

for i in range(10000):
    my_neuron.query([0.5, 0.2])
    my_neuron.train(target=1)

    print(f"Iteration: {i} Weights: {my_neuron._w}, Error: {my_neuron._error}")
