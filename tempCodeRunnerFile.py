import numpy as np
from neuron import Neuron

my_neuron = Neuron(num_inputs=2)
my_neuron.query([0.5, 0.2])

for i in range(300):
    my_neuron.train(target=1)

    print(f"Iteration: {i} Weights: {my_neuron._w}, Error: {my_neuron._error}")