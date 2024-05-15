import math


def relu(x, get_derivative=False):
    return x * (x > 0) if not get_derivative else 1.0 * (x >= 0)


def tanh_act(x, get_derivative=False):
    if not get_derivative:
        return math.tanh(x)  #math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
    return 1 - math.tanh(x) ** 2  #tanh_act(x, get_derivative=False) ** 2


def sigmoid_act(x, get_derivative=False):
    if not get_derivative:
        return 1 / (1 + math.exp(-x))
    return sigmoid_act(x) * (1 - sigmoid_act(x))
