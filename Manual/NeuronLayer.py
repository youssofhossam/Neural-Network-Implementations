import numpy as np
import Neuron

class NeuronLayer:
    def __init__(self,weights, activation_type):
        self.neurons = [] # empty list for upcoming added neurons
    
        for curr_node, prev_nodes in enumerate(weights):
            neuron = Neuron.Neuron(prev_nodes, activation_type) # all the input weigts for the current neuron
            self.neurons.append(neuron)
        
        # for n in self.neurons:
        #     print(n.weights)
            
    def feed_forward(self,input):
        output = [] # out of each neuron after feed forward
        for neuron in self.neurons:
            res = neuron.calc_net_out(input)
            output.append(res)
        # print(output)
        return output