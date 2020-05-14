import numpy as np
from activation_function_utils import activation_functions_utils

np.set_printoptions(suppress=True)


class NeuralNetwork:
    _HIDDEN_LAYER_NEURONS_COUNT = 4

    def __init__(self, training_data_sets, labels, activation_functions=('sigmoid', 'sigmoid')):
        self.training_data_sets = training_data_sets
        self.labels = labels
        self.utils_input_to_layer1 = activation_functions_utils[activation_functions[0]]
        self.weights_input_to_layer1 = np.random.rand(
            self._HIDDEN_LAYER_NEURONS_COUNT,
            self.training_data_sets.shape[1]
        )
        self.utils_layer1_to_output = activation_functions_utils[activation_functions[1]]
        self.weights_layer1_to_output = np.random.rand(labels.shape[1], self._HIDDEN_LAYER_NEURONS_COUNT)
        self.eta = 0.5
        self.output = np.zeros(self.labels.shape)
        self.layer1 = None
        self.ready_for_prediction = False

    def feed_forward(self):
        self.layer1 = self.utils_input_to_layer1.function(
            np.dot(self.training_data_sets, self.weights_input_to_layer1.T)
        )
        self.output = self.utils_layer1_to_output.function(np.dot(self.layer1, self.weights_layer1_to_output.T))

    def back_propagation(self):
        error_layer1_to_out = (self.labels - self.output) * self.utils_layer1_to_output.derivative(self.output)
        weights_layer1_to_output_change = self.eta * np.dot(error_layer1_to_out.T, self.layer1)

        error_input_to_layer1 = \
            self.utils_input_to_layer1.derivative(self.layer1) * \
            np.dot(error_layer1_to_out, self.weights_layer1_to_output)
        weights_input_to_layer1_change = self.eta * np.dot(error_input_to_layer1.T, self.training_data_sets)

        self.weights_input_to_layer1 += weights_input_to_layer1_change
        self.weights_layer1_to_output += weights_layer1_to_output_change

    def learn(self, iterations):
        for _ in range(iterations):
            self.feed_forward()
            self.back_propagation()
        self.ready_for_prediction = True

    def predict(self, input_data_set):
        if not self.ready_for_prediction:
            raise RuntimeError('Neural network must learn before it can make any predictions')
        layer1 = self.utils_input_to_layer1.function(np.dot(input_data_set, self.weights_input_to_layer1.T))
        output = self.utils_layer1_to_output.function(np.dot(layer1, self.weights_layer1_to_output.T))
        return output


def test_activation_functions(training_data_sets, labels, learning_iterations):
    networks = {
        'nn_ss': NeuralNetwork(training_data_sets, labels, activation_functions=('sigmoid', 'sigmoid')),
        'nn_sr': NeuralNetwork(training_data_sets, labels, activation_functions=('sigmoid', 'relu')),
        'nn_rs': NeuralNetwork(training_data_sets, labels, activation_functions=('relu', 'sigmoid')),
        'nn_rr': NeuralNetwork(training_data_sets, labels, activation_functions=('relu', 'relu')),
    }
    for name, network in networks.items():
        network.learn(learning_iterations)
        error = 0.5 * abs(np.sum(labels - network.output))
        print(f'Cost function value for neural network "{name}": {error}')
        print(network.output)


def default_test(training_data_sets, labels, test_title=''):
    print(test_title)
    nn = NeuralNetwork(training_data_sets, labels)
    nn.learn(15000)
    # print(nn.output)
    # the same data s in training data set
    print(nn.predict([[0, 0, 1],
                      [0, 1, 1],
                      [1, 0, 1],
                      [1, 1, 1]]))
    # new previously unknown dataset
    print(nn.predict([[0, 0, 0],
                      [1, 1, 0],
                      [0, 1, 0],
                      [1, 0, 0]]))
    print('---')


if __name__ == '__main__':
    xor_training_data_set = np.array([[0, 0, 1],
                                      [0, 1, 1],
                                      [1, 0, 1],
                                      [1, 1, 1]])
    xor_labels_ = np.array([[1], [0], [0], [1]])

    and_training_data_set = np.array([[1, 1, 1],
                                      [1, 0, 0],
                                      [0, 1, 0],
                                      [0, 0, 0]])
    and_labels_ = np.array([[1], [0], [0], [0]])

    or_training_data_set = np.array([[0, 0, 0],
                                     [0, 1, 1],
                                     [1, 0, 1],
                                     [1, 1, 1]])
    or_labels_ = np.array([[0], [1], [1], [1]])

    print('--- DEFAULT TEST ---')
    default_test(xor_training_data_set, xor_labels_, 'xor')
    default_test(and_training_data_set, and_labels_, 'and')
    default_test(or_training_data_set, or_labels_, 'or')

    print('--- ACTIVATION FUNCTIONS TEST ---')
    test_activation_functions(xor_training_data_set, xor_labels_, 15000)
