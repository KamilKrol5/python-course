import numpy as np

from neural_network import NeuralNetwork, NeuralNetworkHiddenLayerInfo
from task2_function_approximation import learn_function


def learn_parabolic():
    training_data = np.linspace(-50, 50, 26)
    labels = training_data ** 2
    test_data = np.linspace(-50, 50, 101)
    expected_from_test = test_data ** 2
    nn = NeuralNetwork(
        [NeuralNetworkHiddenLayerInfo('sigmoid', 1),
         NeuralNetworkHiddenLayerInfo('relu', 10),
         NeuralNetworkHiddenLayerInfo('sigmoid', 10)],
        1
    )
    learn_function(training_data, labels, test_data, expected_from_test, nn)


def learn_sine():
    training_data = np.linspace(0, 2, 21)
    labels = np.sin((3 * np.pi / 2) * training_data)
    test_data = np.linspace(0, 2, 161)
    expected_from_test = np.sin((3 * np.pi / 2) * test_data)
    nn = NeuralNetwork(
        [NeuralNetworkHiddenLayerInfo('tanh', 1),
         NeuralNetworkHiddenLayerInfo('tanh', 10),
         NeuralNetworkHiddenLayerInfo('sigmoid', 10)],
        1
    )
    learn_function(training_data, labels, test_data, expected_from_test, nn)


if __name__ == '__main__':
    # Answer:
    # For this cases (1-10-10-1 neural network) the results are far better than in the task2.
    # However, the activation functions have significant impact on the results.
    # When the activation functions are inappropriate the results can be very poor.
    learn_parabolic()
    learn_sine()
