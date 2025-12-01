import NeuronLayer
import numpy as np

class NeuralNetwork:
    def __init__(self, hidden_layer_weights, output_layer_weights, activation_type = 'poly'):
        self.hidden_layer = NeuronLayer.NeuronLayer(hidden_layer_weights, activation_type)
        self.output_layer = NeuronLayer.NeuronLayer(output_layer_weights,activation_type)
        self.learning_rate = .5
    def feed_forward(self, input):
        self.hidden_layer_out = self.hidden_layer.feed_forward(input)
        self.output_layer_out = self.output_layer.feed_forward(self.hidden_layer_out)
        return self.output_layer_out
    
    def compute_error(self , target):
        # output layer
        self.de_do_net = np.zeros(len(self.output_layer.neurons))
        for i, o_neuron in enumerate(self.output_layer.neurons):
            self.de_do_out = o_neuron.out - target[i]
            self.do_out_do_net = o_neuron.activation_derv()
            self.de_do_net[i] = self.de_do_out * self.do_out_do_net
            print(self.de_do_net[i])
        # hidden layer
        self.de_dh_net = np.zeros(len(self.hidden_layer.neurons)) 
        for h, h_neuron in enumerate(self.hidden_layer.neurons):
            self.do_net_dh_out = 0
            for o, o_neuron in enumerate(self.output_layer.neurons):
                o_weights = o_neuron.weights[h]
                self.do_net_dh_out += o_weights * self.de_do_net[o]
            self.dh_out_dh_net = h_neuron.activation_derv()
            self.de_dh_net[h] = self.dh_out_dh_net * self.do_net_dh_out
            print(self.de_dh_net[h])
    def update_weights(self):
        # output layer
        for o, o_neuron in enumerate(self.output_layer.neurons):
            for w, weight in enumerate(o_neuron.weights):
                de_dw = self.de_do_net[o] * o_neuron.input[w]
                weight -= de_dw * self.learning_rate
            print(weight)
        for h, h_neuron in enumerate(self.hidden_layer.neurons):
            for w,weight in enumerate(h_neuron.weights):
                de_dw = self.de_dh_net[h] * h_neuron.input[w]
                weight -= de_dw * self.learning_rate
            print(weight)

    def train(self,input,target):
        o_out = self.feed_forward(input)
        self.compute_error(target)
        self.update_weights()