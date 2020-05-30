import numpy as np

from activation_function_utils import activation_functions_utils
from neural_network import NeuralNetwork, NeuralNetworkHiddenLayerInfo

np.set_printoptions(suppress=True)

networks2 = {
    '2 sigmoid sigmoid': NeuralNetwork([
        NeuralNetworkHiddenLayerInfo('sigmoid', 2),
        NeuralNetworkHiddenLayerInfo('sigmoid', 4)], 1, eta=0.4),
    '2 sigmoid relu': NeuralNetwork([
        NeuralNetworkHiddenLayerInfo('sigmoid', 2),
        NeuralNetworkHiddenLayerInfo('relu', 4)], 1, eta=0.4),
    '2 relu sigmoid': NeuralNetwork([
        NeuralNetworkHiddenLayerInfo('relu', 2),
        NeuralNetworkHiddenLayerInfo('sigmoid', 4)], 1, eta=0.4),
    '2 relu relu': NeuralNetwork([
        NeuralNetworkHiddenLayerInfo('relu', 2),
        NeuralNetworkHiddenLayerInfo('relu', 4)], 1, eta=0.4),
}

networks3 = {
    '3 sigmoid sigmoid': NeuralNetwork([
        NeuralNetworkHiddenLayerInfo('sigmoid', 3),
        NeuralNetworkHiddenLayerInfo('sigmoid', 4)], 1, eta=0.4),
    '3 sigmoid relu': NeuralNetwork([
        NeuralNetworkHiddenLayerInfo('sigmoid', 3),
        NeuralNetworkHiddenLayerInfo('relu', 4)], 1, eta=0.4),
    '3 relu sigmoid': NeuralNetwork([
        NeuralNetworkHiddenLayerInfo('relu', 3),
        NeuralNetworkHiddenLayerInfo('sigmoid', 4)], 1, eta=0.4),
    '3 relu relu': NeuralNetwork([
        NeuralNetworkHiddenLayerInfo('relu', 3),
        NeuralNetworkHiddenLayerInfo('relu', 4)], 1, eta=0.4),
}


def test_activation_functions(
        training_data_sets,
        labels,
        learning_iterations,
        networks,
        test_data_sets=None,
        test_labels=None
) -> None:
    for name, network in networks.items():
        network.learn(training_data_sets, labels, learning_iterations)
        error = np.square(labels - network.output).mean()
        print(f'Cost function value for TRAINING data for:\n"{name}": {error}')
        print(f'Got: {", ".join(str(x[0]) for x in network.output)}, '
              f'expected: {", ".join(str(x[0]) for x in labels)}\n')

        if training_data_sets is not None and test_labels is not None:
            network.predict(test_data_sets)
            error2 = np.square(test_labels - network.output).mean()
            print(f'Cost function value for TEST data for:\n "{name}": {error2}')
            print(f'Got: {", ".join(str(x[0]) for x in network.output)}, '
                  f'expected: {", ".join(str(x[0]) for x in test_labels)}')
            print('---')


if __name__ == '__main__':
    data3 = {
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

    data2 = {
        'xor': {
            'training': np.array([[0, 0],
                                  [0, 1],
                                  [1, 0],
                                  [1, 1]]),
            'training_labels': np.array([[0], [1], [1], [0]]),
        },
        'and': {
            'training': np.array([[1, 1],
                                  [1, 0],
                                  [0, 1],
                                  [0, 0]]),
            'training_labels': np.array([[1], [0], [0], [0]]),
        },
        'or': {
            'training': np.array([[0, 0],
                                  [0, 1],
                                  [1, 0],
                                  [1, 1]]),
            'training_labels': np.array([[0], [1], [1], [1]]),
        }
    }

    print('--- ACTIVATION FUNCTIONS TEST: 2 variables---')
    print('---------------------------------------------------------XOR:')
    test_activation_functions(
        data2['xor']['training'],
        data2['xor']['training_labels'],
        15000,
        networks2
    )

    print('---------------------------------------------------------AND:')
    test_activation_functions(
        data2['and']['training'],
        data2['and']['training_labels'],
        15000,
        networks2
    )

    print('---------------------------------------------------------OR:')
    test_activation_functions(
        data2['or']['training'],
        data2['or']['training_labels'],
        15000,
        networks2
    )
    print('--- ACTIVATION FUNCTIONS TEST: 3 variables---')
    print('---------------------------------------------------------XOR:')
    test_activation_functions(
        data3['xor']['training'],
        data3['xor']['training_labels'],
        15000,
        networks3,
        data3['xor']['test'],
        data3['xor']['test_labels'],
    )

    print('---------------------------------------------------------AND:')
    test_activation_functions(
        data3['and']['training'],
        data3['and']['training_labels'],
        15000,
        networks3,
        data3['and']['test'],
        data3['and']['test_labels'],
    )

    print('---------------------------------------------------------OR:')
    test_activation_functions(
        data3['or']['training'],
        data3['or']['training_labels'],
        15000,
        networks3,
        data3['or']['test'],
        data3['or']['test_labels'],
    )
