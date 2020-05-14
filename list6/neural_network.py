from dataclasses import dataclass
from typing import List

import numpy as np

from activation_function_utils import activation_functions_utils, ActivationFunctionUtils

np.set_printoptions(suppress=True)


@dataclass
class NeuralNetworkHiddenLayer:
    activation_function_name: str = None
    neurons_count: int = 5
    weights: np.ndarray = None
    utils: ActivationFunctionUtils = None
    values: np.ndarray = None
    weights_change: np.ndarray = None

    def update_weights(self):
        self.weights += self.weights_change
        self.weights_change.fill(0)


class NeuralNetwork:
    def __init__(self,
                 training_data_sets: np.ndarray,
                 labels: np.ndarray,
                 hidden_layers: List[NeuralNetworkHiddenLayer],
                 ):
        if len(hidden_layers) < 2:
            raise NotImplementedError('This implementation of neural network does not support single layer networks')

        self.hidden_layers = hidden_layers
        self.input_layer_neuron_count = self.hidden_layers[0].neurons_count
        self.training_data_sets = training_data_sets
        self.labels = labels

        for layer, next_layer in zip(self.hidden_layers[0:-1], self.hidden_layers[1:]):
            layer.utils = activation_functions_utils[layer.activation_function_name]
            layer.weights = np.random.rand(
                next_layer.neurons_count,
                layer.neurons_count,
            )
            layer.values = None

        last_hidden_layer = self.hidden_layers[-1]
        last_hidden_layer.utils = activation_functions_utils[last_hidden_layer.activation_function_name]
        last_hidden_layer.weights = np.random.rand(labels.shape[1], last_hidden_layer.neurons_count)
        self.output = np.zeros(self.labels.shape)

        self.eta = 0.5
        self.ready_for_prediction = False

    def _feed_forward(self, with_training_data=True):
        input_layer = self.hidden_layers[0]
        if with_training_data:
            input_layer.values = self.training_data_sets
        for layer, next_layer in zip(self.hidden_layers[0:-1], self.hidden_layers[1:]):
            next_layer.values = layer.utils.function(
                np.dot(layer.values, layer.weights.T)
            )
        last_hidden_layer = self.hidden_layers[-1]
        self.output = last_hidden_layer.utils.function(
            np.dot(last_hidden_layer.values, last_hidden_layer.weights.T)
        )

    def _back_propagation(self):
        last_hidden_layer = self.hidden_layers[-1]
        error = (self.labels - self.output) * last_hidden_layer.utils.derivative(self.output)
        last_hidden_layer.weights_change = self.eta * np.dot(error.T, last_hidden_layer.values)

        for layer, next_layer in zip(reversed(self.hidden_layers[0:-1]), reversed(self.hidden_layers[1:])):
            error = layer.utils.derivative(next_layer.values) * \
                    np.dot(error, next_layer.weights)
            layer.weights_change = self.eta * np.dot(error.T, layer.values)

        for layer in self.hidden_layers:
            layer.update_weights()

    def learn(self, iterations):
        for _ in range(iterations):
            self.hidden_layers[0].values = self.training_data_sets
            self._feed_forward()
            self._back_propagation()
        self.ready_for_prediction = True

    def predict(self, input_data_set):
        if not self.ready_for_prediction:
            raise RuntimeError('Neural network must learn before it can make any predictions')
        input_layer = self.hidden_layers[0]
        input_layer.values = input_data_set
        self._feed_forward(with_training_data=False)
        return self.output
