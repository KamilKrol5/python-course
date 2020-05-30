from typing import List

import numpy as np

from activation_function_utils import (
    activation_functions_utils,
    ActivationFunctionUtils,
)

np.set_printoptions(suppress=True)


class NeuralNetworkHiddenLayerInfo:
    def __init__(self, activation_function_name: str, neurons_count: int):
        self.activation_function_name: str = activation_function_name
        self.neurons_count: int = neurons_count


class NeuralNetworkHiddenLayer:
    def __init__(self, basic_info: NeuralNetworkHiddenLayerInfo, weights: np.ndarray, biases: np.ndarray):
        self.info = basic_info
        self.utils: ActivationFunctionUtils = activation_functions_utils[
            basic_info.activation_function_name
        ]
        self.weights: np.ndarray = weights
        self.values: np.ndarray = np.zeros(shape=(self.info.neurons_count,))
        self.biases = biases


class NeuralNetwork:
    """
    Class representing a neural network (multilayer perceptron).
    """

    def __init__(
            self,
            hidden_layers_info: List[NeuralNetworkHiddenLayerInfo],
            output_neurons_count,
            eta=0.02
    ):
        """
        Args:
            hidden_layers_info (List[NeuralNetworkHiddenLayerInfo): A list of objects
                with info needed for creation of hidden layers. There must be at least two layers,
                since single layer network is not supported. The first layer is input layer.
            output_neurons_count (int): The number of neurons in output layer.
        """
        if len(hidden_layers_info) < 2:
            raise NotImplementedError(
                "This implementation of neural network does not support single layer networks"
            )
        self.output_neurons_count = output_neurons_count
        self.output = np.zeros((output_neurons_count, 1))

        self.hidden_layers_info = hidden_layers_info
        self.input_layer_neuron_count = self.hidden_layers_info[0].neurons_count
        self.hidden_layers = self._create_hidden_layers()

        self.eta = eta

    def feed_forward(
            self,
            training_data_sets: np.ndarray,
    ) -> None:
        """
        Feeds the neural network with the data provided.

        Args:
            training_data_sets (numpy.ndarray): Data which will be used for feeding the network.
        """
        input_layer = self.hidden_layers[0]
        input_layer.values = training_data_sets
        for layer, next_layer in zip(self.hidden_layers[0:-1], self.hidden_layers[1:]):
            next_layer.values = layer.utils.function(
                np.dot(layer.values, layer.weights.T) + layer.biases.reshape(1, -1)
            )
        last_hidden_layer = self.hidden_layers[-1]
        self.output = last_hidden_layer.utils.function(
            np.dot(last_hidden_layer.values, last_hidden_layer.weights.T)
        )
        # input_layer = self.hidden_layers[0]
        # # print('out=', self.output)
        # # print('l1', self.hidden_layers[0].values)
        # # print('l5', self.hidden_layers[1].values)
        # input_layer.values = training_data_sets
        # assert input_layer.info.neurons_count == 1
        # self.hidden_layers[1].values = input_layer.utils.function(
        #     np.dot(training_data_sets, input_layer.weights.T))
        # self.output = self.hidden_layers[1].utils.function(
        #     np.dot(self.hidden_layers[1].values, self.hidden_layers[1].weights.T))
        # print('out after=', self.output)

    def back_propagation(
            self,
            labels: np.ndarray
    ) -> None:
        """
        Method which runs backward propagation algorithm.

        Args:
            labels (numpy.ndarray): Data which will be used for execution of backward propagation algorithm.
                It's dimensions must be consistent with training data sets and first layer's neuron count.
        """
        deltas = []
        bias_deltas = []
        last_hidden_layer = self.hidden_layers[-1]
        error = (labels - self.output) * last_hidden_layer.utils.derivative(self.output)
        deltas.append(self.eta * np.dot(error.T, last_hidden_layer.values))
        bias_deltas.append(self.eta * np.sum(error, axis=0, keepdims=True))

        for layer, next_layer in zip(
                reversed(self.hidden_layers[0:-1]), reversed(self.hidden_layers[1:])
        ):
            error = layer.utils.derivative(next_layer.values) * np.dot(
                error, next_layer.weights
            )
            deltas.append(self.eta * np.dot(error.T, layer.values))
            bias_deltas.append(self.eta * np.sum(error, axis=0, keepdims=True))

        for layer, weights_change, bias_change in zip(self.hidden_layers, reversed(deltas), reversed(bias_deltas)):
            layer.weights += weights_change
            layer.biases += bias_change.reshape(-1)
        # last_layer = self.hidden_layers[-1]
        # delta2 = (labels - self.output) * \
        #          last_layer.utils.derivative(self.output)
        # d_weights2 = self.eta * np.dot(delta2.T, last_layer.values)
        #
        # delta1 = self.hidden_layers[0].utils.derivative(last_layer.values) * \
        #          np.dot(delta2, last_layer.weights)
        # d_weights1 = self.eta * np.dot(delta1.T, self.hidden_layers[0].values)
        #
        # self.hidden_layers[0].weights += d_weights1
        # self.hidden_layers[1].weights += d_weights2

    def _create_hidden_layers(self) -> List[NeuralNetworkHiddenLayer]:
        hidden_layers = []
        for layer_info, next_layer_info in zip(
                self.hidden_layers_info[0:-1], self.hidden_layers_info[1:]
        ):
            weights = np.random.standard_normal(
                (next_layer_info.neurons_count, layer_info.neurons_count)
            )
            biases = np.random.rand(next_layer_info.neurons_count)
            hidden_layers.append(NeuralNetworkHiddenLayer(layer_info, weights, biases))

        last_hidden_layer_info = self.hidden_layers_info[-1]
        last_hidden_layer_info_weights = np.random.standard_normal(
            (self.output.shape[1], last_hidden_layer_info.neurons_count)
        )
        last_hidden_layer_biases = np.random.rand(self.output_neurons_count)
        hidden_layers.append(
            NeuralNetworkHiddenLayer(
                last_hidden_layer_info, last_hidden_layer_info_weights, last_hidden_layer_biases
            )
        )
        return hidden_layers

    def learn(
            self,
            training_data_sets: np.ndarray,
            labels: np.ndarray,
            iterations: int,
    ) -> None:
        """
        Method which trains the network using gradient descent and backward propagation methods.

        Args:
            training_data_sets (numpy.ndarray): Data which will be used for training. If not provided,
                there is up to user to set up weights and biases to make any predictions 'sensible'
            labels (numpy.ndarray): same as for training_data_sets. It's dimensions must be consistent with
                training data sets and first layer's neuron count.
            iterations (int): Number of iterations that should be performed
                when learning the network with provided data.
        """
        if training_data_sets is None:
            raise RuntimeError("The network cannot learn when learning data is not provided.")

        self.hidden_layers[0].values = training_data_sets
        for i in range(iterations):
            self.feed_forward(training_data_sets)
            self.back_propagation(labels)

    def predict(self, input_data_set: np.ndarray) -> np.ndarray:
        self.feed_forward(input_data_set)
        return self.output
