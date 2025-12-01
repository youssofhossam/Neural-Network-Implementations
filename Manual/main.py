import numpy as np
import NeuralNetwork
def poly():
    hidden_layer_weights = np.array([[1, 1],
                                     [2, 1]])
    output_layer_weights = np.array([[2, 1],
                                     [1, 0]])
    nn = NeuralNetwork(hidden_layer_weights,output_layer_weights)
    nn.train([1, 1], [290, 14])
    # print('hello')


def sigm():     # 2 4 3
    hidden_layer_weights = np.array([[0.1, 0.1],      
                                     [0.2, 0.1],
                                     [0.1, 0.3],
                                     [0.5, 0.01]])

    output_layer_weights = np.array([[0.1, 0.2, 0.1, 0.2],
                                     [0.1, 0.1, 0.1, 0.5],
                                     [0.1, 0.4, 0.3, 0.2]])

    nn = NeuralNetwork.NeuralNetwork(hidden_layer_weights, output_layer_weights, 'sigm')

    nn.train([1, 2], [0.4, 0.7, 0.6])


if __name__ == '__main__':
    # poly()
    sigm()