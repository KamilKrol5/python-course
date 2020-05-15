from collections import namedtuple
from scipy.special import expit as sigmoid
import numpy as np


def _sigmoid_star(x):
    return x * (1.0 - x)


def _relu(x):
    return np.maximum(0, x)


def _relu_gradient(x):
    return np.where(x > 0, 1, 0)


def tanh_derivative(x):
    return 1.0 - np.tanh(x)**2


ActivationFunctionUtils = namedtuple('_ActivationFunctionUtils', ['function', 'derivative'])

activation_functions_utils = {
    'sigmoid': ActivationFunctionUtils(function=sigmoid, derivative=_sigmoid_star),
    'relu': ActivationFunctionUtils(function=_relu, derivative=_relu_gradient),
    'tanh': ActivationFunctionUtils(function=np.tanh, derivative=tanh_derivative)
}
