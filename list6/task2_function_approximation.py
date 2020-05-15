import numpy as np
import matplotlib.pyplot as plt

from neural_network import NeuralNetwork, NeuralNetworkHiddenLayer


def scale_data(data: np.ndarray, target_range=(0, 1)) -> np.ndarray:
    max_, min_ = np.max(data), np.min(data)
    length = max_ - min_
    target_length = target_range[1] - target_range[0]
    return target_length / length * (data - min_) + target_range[0]


def test_scale_data():
    t1 = np.linspace(-50, -45, 5)
    print(scale_data(t1, (45, 50)))
    print(scale_data(t1, (0, 1)))
    t2 = np.linspace(2, 4, 8)
    print(scale_data(t2, (0, 1)))
    print(scale_data(t2, (10, 90)))


def learn_parabolic():
    training_data = np.linspace(-50, 50, 26)
    labels = training_data ** 2
    plt.scatter(training_data.reshape((-1, 1)), labels.reshape((-1, 1)))
    plt.title('y=x^2')
    plt.show()

    test_data = np.linspace(-50, 50, 101)
    expected_from_test = test_data ** 2
    nn = NeuralNetwork(scale_data(test_data, (-1, 1)).reshape((-1, 1)),
                       (scale_data(expected_from_test, (-1, 1))).reshape((-1, 1)),
                       [NeuralNetworkHiddenLayer('sigmoid', 1), NeuralNetworkHiddenLayer('sigmoid', 5)]
                       )
    nn.learn(8000, draw=1000)


def learn_sinus():
    training_data = np.linspace(0, 2, 21)
    labels = np.sin((3 * np.pi / 2) * training_data)
    plt.scatter(training_data, labels)
    plt.title('y=sin(3PI * x / 3)')
    plt.show()

    test_data = np.linspace(0, 2, 161)
    expected_from_test = np.sin((3 * np.pi / 2) * test_data)
    nn = NeuralNetwork(scale_data(test_data).reshape((-1, 1)),
                       scale_data(expected_from_test).reshape((-1, 1)),
                       [NeuralNetworkHiddenLayer('sigmoid', 1), NeuralNetworkHiddenLayer('sigmoid', 5)]
                       )
    nn.learn(10000, draw=1000)


if __name__ == '__main__':

    learn_parabolic()
    learn_sinus()

