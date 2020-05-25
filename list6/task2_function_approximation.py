import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from neural_network import NeuralNetwork, NeuralNetworkHiddenLayerInfo


def scale_data(data: np.ndarray, target_range=(0, 1)) -> np.ndarray:
    # max_, min_ = np.max(data), np.min(data)
    # length = max_ - min_
    # target_length = target_range[1] - target_range[0]
    # return target_length / length * (data - min_) + target_range[0]
    scaler = MinMaxScaler(target_range)
    return scaler.fit_transform(data)


def test_scale_data():
    t1 = np.linspace(-50, -45, 5)
    print(scale_data(t1, (45, 50)))
    print(scale_data(t1, (0, 1)))
    t2 = np.linspace(2, 4, 8)
    print(scale_data(t2, (0, 1)))
    print(scale_data(t2, (10, 90)))
    t3 = np.linspace(-50, 50, 26)
    print(scale_data(t3, (0, 1)))


def learn_function(training_data_x, training_data_y, test_data_x, test_data_y, nn: NeuralNetwork=None):
    training_data_x = training_data_x.reshape((-1, 1))
    training_data_y = training_data_y.reshape((-1, 1))
    test_data_x = test_data_x.reshape((-1, 1))
    test_data_y = test_data_y.reshape((-1, 1))
    # build neural network
    if nn is None:
        nn = NeuralNetwork(
            [NeuralNetworkHiddenLayerInfo('relu', 1),
             NeuralNetworkHiddenLayerInfo('tanh', 50),
             NeuralNetworkHiddenLayerInfo('sigmoid', 20)],
            1
        )
    # preprocess data for nn
    scaler_x = MinMaxScaler((0, 1))
    scaler_y = MinMaxScaler((0, 1))
    scaled_x_train = scaler_x.fit_transform(training_data_x)
    scaled_y_train = scaler_y.fit_transform(training_data_y)
    scaled_x_test = scaler_x.fit_transform(test_data_x)
    # learning
    draw = 10000
    for i in range(1000000):
        nn.feed_forward(scaled_x_train)
        nn.back_propagation(scaled_y_train)

        # drawing for test data
        if i % draw == 0:
            pr = nn.predict(scaled_x_test)
            plt.scatter(test_data_x, test_data_y, 14)
            plt.scatter(test_data_x, scaler_y.inverse_transform(pr), 14)
            plt.title(f'Iteration: {i} '
                      f'Error: {np.square(test_data_y - scaler_y.inverse_transform(nn.output)).mean()}')
            plt.show()

    plt.scatter(test_data_x, test_data_y)
    plt.title(f'Function')
    plt.show()

    pr = nn.predict(scaled_x_test)
    plt.scatter(test_data_x, scaler_y.inverse_transform(pr))
    plt.title(f'End, Error: '
              f'{np.square(test_data_y - scaler_y.inverse_transform(pr)).mean()}')
    plt.show()


def learn_parabolic():
    training_data = np.linspace(-50, 50, 26)
    labels = training_data ** 2
    test_data = np.linspace(-50, 50, 101)
    expected_from_test = test_data ** 2
    learn_function(training_data, labels, test_data, expected_from_test)


def learn_sine():
    training_data = np.linspace(0, 2, 21)
    labels = np.sin((3 * np.pi / 2) * training_data)
    test_data = np.linspace(0, 2, 101)
    expected_from_test = np.sin((3 * np.pi / 2) * test_data)
    learn_function(training_data, labels, test_data, expected_from_test)


def learn_test():
    test_data = np.linspace(-10, 10, 161)
    expected_from_test = test_data ** 2
    learn_function(test_data, expected_from_test, test_data, expected_from_test)


if __name__ == '__main__':
    # test_scale_data()
    learn_parabolic()
    # learn_sine()
    # learn_test()