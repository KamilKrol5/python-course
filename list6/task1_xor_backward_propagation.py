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


def test_activation_functions(training_data_sets, labels, test_data_sets, test_labels, learning_iterations):
    networks = {
        'sigmoid sigmoid': NeuralNetwork(training_data_sets, labels, activation_functions=('sigmoid', 'sigmoid')),
        'sigmoid relu': NeuralNetwork(training_data_sets, labels, activation_functions=('sigmoid', 'relu')),
        'relu sigmoid': NeuralNetwork(training_data_sets, labels, activation_functions=('relu', 'sigmoid')),
        'relu relu': NeuralNetwork(training_data_sets, labels, activation_functions=('relu', 'relu')),
    }
    for name, network in networks.items():
        network.learn(learning_iterations)
        error = np.square(labels - network.output).mean()
        print(f'Cost function value for TRAINING data for:\n"{name}": {error}')
        print(f'Got: {", ".join(str(x[0]) for x in network.output)}, '
              f'expected: {", ".join(str(x[0]) for x in labels)}\n')

        network.predict(test_data_sets)
        error2 = np.square(test_labels - network.output).mean()
        print(f'Cost function value for TEST data for:\n "{name}": {error2}')
        print(f'Got: {", ".join(str(x[0]) for x in network.output)}, '
              f'expected: {", ".join(str(x[0]) for x in test_labels)}')
        print('---')


def default_test(data_, test_title=''):
    print(test_title)
    nn = NeuralNetwork(data_['training'], data_['training_labels'])
    nn.learn(15000)
    print(nn.predict(data_['training']))
    print(nn.predict(data_['test']))
    print('---')


if __name__ == '__main__':
    data = {
        'xor': {
            'training': np.array([[0, 0, 1],
                                  [0, 1, 1],
                                  [1, 0, 1],
                                  [1, 1, 1]]),
            'training_labels': np.array([[1], [0], [0], [1]]),
            'test': np.array([[0, 0, 0],
                              [1, 1, 0],
                              [0, 1, 0],
                              [1, 0, 0]]),
            'test_labels': np.array([[0], [0], [1], [1]]),
        },
        'and': {
            'training': np.array([[1, 1, 1],
                                  [1, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 0]]),
            'training_labels': np.array([[1], [0], [0], [0]]),
            'test': np.array([[1, 1, 0],
                              [1, 0, 1],
                              [0, 0, 1],
                              [0, 1, 1]]),
            'test_labels': np.array([[0], [0], [0], [0]]),
        },
        'or': {
            'training': np.array([[0, 0, 0],
                                  [0, 1, 1],
                                  [1, 0, 1],
                                  [1, 1, 1]]),
            'training_labels': np.array([[0], [1], [1], [1]]),
            'test': np.array([[0, 0, 1],
                              [0, 1, 0],
                              [1, 1, 0],
                              [1, 0, 0]]),
            'test_labels': np.array([[1], [1], [1], [1]]),
        }
    }

    # print('--- DEFAULT TEST ---')
    # default_test(data['xor'], 'xor')
    # default_test(data['and'], 'and')
    # default_test(data['or'], 'or')

    print('--- ACTIVATION FUNCTIONS TEST ---')
    print('---------------------------------------------------------XOR:')
    test_activation_functions(
        data['xor']['training'],
        data['xor']['training_labels'],
        data['xor']['test'],
        data['xor']['test_labels'],
        15000,
    )

    print('---------------------------------------------------------AND:')
    test_activation_functions(
        data['and']['training'],
        data['and']['training_labels'],
        data['and']['test'],
        data['and']['test_labels'],
        15000,
    )

    print('---------------------------------------------------------OR:')
    test_activation_functions(
        data['or']['training'],
        data['or']['training_labels'],
        data['or']['test'],
        data['or']['test_labels'],
        15000,
    )
