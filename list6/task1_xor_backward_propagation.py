import numpy as np
from scipy.special import expit as sigmoid


def _sigmoid_gradient(x):
    fx = sigmoid(x)
    return fx * (1.0 - fx)


class NeuralNetwork:
    _HIDDEN_LAYER_NEURONS_COUNT = 4

    def __init__(self, training_data_sets, labels):
        self.training_data_sets = training_data_sets
        self.labels = labels
        self.weights_input_to_layer1 = np.random.rand(
            self._HIDDEN_LAYER_NEURONS_COUNT,
            self.training_data_sets.shape[1]
        )
        self.weights_layer1_to_output = np.random.rand(labels.shape[1], self._HIDDEN_LAYER_NEURONS_COUNT)
        self.eta = 0.5
        self.output = np.zeros(self.labels.shape)
        self.layer1 = None
        self.ready_for_prediction = False

    def feed_forward(self):
        self.layer1 = sigmoid(np.dot(self.training_data_sets, self.weights_input_to_layer1.T))
        self.output = sigmoid(np.dot(self.layer1, self.weights_layer1_to_output.T))

    def back_propagation(self):
        error_layer1_to_out = (self.labels - self.output) * _sigmoid_gradient(self.output)
        weights_layer1_to_output_change = self.eta * np.dot(error_layer1_to_out.T, self.layer1)

        error_input_to_layer1 = \
            _sigmoid_gradient(self.layer1) * np.dot(error_layer1_to_out, self.weights_layer1_to_output)
        weights_input_to_layer1_change = self.eta * np.dot(error_input_to_layer1.T, self.training_data_sets)

        self.weights_input_to_layer1 += weights_input_to_layer1_change
        self.weights_layer1_to_output += weights_layer1_to_output_change

    def learn(self, iterations):
        for _ in range(iterations):
            self.feed_forward()
            self.back_propagation()
        self.ready_for_prediction = True

    def predict(self, data_set):
        if not self.ready_for_prediction:
            raise RuntimeError('Neural network must learn before it can make any predictions')
        layer1 = sigmoid(np.dot(data_set, self.weights_input_to_layer1.T))
        output = sigmoid(np.dot(layer1, self.weights_layer1_to_output.T))
        return output


if __name__ == '__main__':
    training_data_sets_ = np.array([[0, 0, 1],
                                    [0, 1, 1],
                                    [1, 0, 1],
                                    [1, 1, 1]])
    labels_ = np.array([[0], [1], [1], [0]])
    nn = NeuralNetwork(training_data_sets_, labels_)
    nn.learn(15000)
    print(nn.output)

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
