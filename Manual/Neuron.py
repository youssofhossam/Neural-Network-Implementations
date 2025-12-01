import math
import numpy as np

class Neuron:
    def __init__(self,weights, activation_type):
        self.weights = weights
        self.activation_type = activation_type
        self.out = None
        self.net = None
        self.input = None

    def calc_net_out(self, input):
        self.input = input
        self.net = np.dot(self.weights, self.input)
        self.out = self.activation(self.net)
        return self.out
    def activation(self,net):
        if(self.activation_type == 'sigm'):
            return 1 / (1 + math.exp(-net))
        elif(self.activation_type == 'poly'):
            return net ** 2
    def activation_derv(self):
        if(self.activation_type == 'sigm'):
            return math.exp(-self.net) / math.pow(1 + math.exp(-self.net) ,2)
        elif(self.activation_type == 'poly'):
            return self.net * 2
        