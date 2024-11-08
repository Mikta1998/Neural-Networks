import numpy as np

def sigmoid(x:float):
    return 1/(1+np.exp(-x))

def dsigmoid(last_output: float):
    return last_output * (1.0 - last_output)