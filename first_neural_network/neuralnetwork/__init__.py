import random
import math
from .neuronlayer import NeuronLayer

class NeuralNetwork:

    def __init__(self, learning_rate, inputs_number, hidden_number, outputs_number, weights_service):
        self.learning_rate = learning_rate
        self.inputs_number = inputs_number

        self.hidden_layer = NeuronLayer(hidden_number, inputs_number,  weights_service)
        self.output_layer = NeuronLayer(outputs_number, hidden_number,  weights_service)

    def feed_forward(self, inputs):
        ''' sends the input through the network returning the inferred output '''
        hidden_layer_outputs = self.hidden_layer.feed_forward(inputs)
        return self.output_layer.feed_forward(hidden_layer_outputs)

    def compute_output_deltas(self,training_outputs):
        ''' for each output unit computes its delta '''
        output_deltas = [0] * len(self.output_layer.neurons)
        for k, output_nu in enumerate(self.output_layer.neurons):

            output_deltas[k] = output_nu.delta(training_outputs[k])
        
        return output_deltas

    def compute_hidden_deltas(self,output_deltas):
        ''' for each hidden unit computes its delta using backprop '''
        hidden_deltas = [0] * len(self.hidden_layer.neurons)
        for j , hidden_nu in enumerate(self.hidden_layer.neurons):

            summation_of_deltas = 0
            for k in range(len(self.output_layer.neurons)):
                summation_of_deltas += output_deltas[k] * self.output_layer.neurons[k].weights[j]

            hidden_deltas[j] = summation_of_deltas * hidden_nu.dout_dnet()
        
        return hidden_deltas

    def weights_iterator(self,nnlayer):
        ''' python generator cycling through the weights of a given layer '''
        for n, neuron_unit in enumerate(nnlayer.neurons):
            for weight_index in range(len(neuron_unit.weights)):
                yield n, neuron_unit, weight_index

    def online_training(self, training_set):
        ''' performing online training, the training set is shuffled at the beginning'''
        random.shuffle(training_set)
        for pattern in training_set:
            training_input, training_output = pattern
            self.feed_forward(training_input)

            # 1. Output neuron deltas
            output_deltas = self.compute_output_deltas(training_output)

            # 2. Hidden neuron deltas
            hidden_deltas = self.compute_hidden_deltas(output_deltas)

            # 3. Update output neuron weights
            for k, output_unit, weight_index in self.weights_iterator(self.output_layer):
                output_unit.weights[weight_index] -= self.learning_rate * output_deltas[k] * output_unit.dnet_dwj(weight_index)

            # 4. Update hidden neuron weights
            for j, hidden_unit, weight_index in self.weights_iterator(self.hidden_layer):
                hidden_unit.weights[weight_index] -= self.learning_rate * hidden_deltas[j] * hidden_unit.dnet_dwj(weight_index)

    def batch_training(self, training_set):
        ''' performing batch training'''
        outputweights_matrix = [ [ 0 for i in range(len(self.hidden_layer.neurons)) ] for j in range(len(self.output_layer.neurons)) ]
        hiddenweights_matrix = [ [ 0 for i in range(self.inputs_number) ] for j in range(len(self.hidden_layer.neurons)) ]
        for p_index, pattern in enumerate(training_set):
            input_pattern, output_pattern = pattern
            self.feed_forward(input_pattern)
            # 1. Output neuron deltas
            output_deltas = self.compute_output_deltas(output_pattern)
            # 2. Hidden neuron deltas
            hidden_deltas = self.compute_hidden_deltas(output_deltas)

            for k, output_nu, weight_index in self.weights_iterator(self.output_layer):
                outputweights_matrix[k][weight_index] += output_deltas[k] * output_nu.dnet_dwj(weight_index)

            for j, hidden_nu, weight_index in self.weights_iterator(self.hidden_layer):
                hiddenweights_matrix[j][weight_index] += hidden_deltas[j] * hidden_nu.dnet_dwj(weight_index)
        
        # 3. Update output neuron weights
        for k, output_nu, weight_index in self.weights_iterator(self.output_layer):
            output_nu.weights[weight_index] -= self.learning_rate * outputweights_matrix[k][weight_index]
        
        # 4. Update hidden neuron weights
        for j, hidden_nu, weight_index in self.weights_iterator(self.hidden_layer):
            hidden_nu.weights[weight_index] -= self.learning_rate * hiddenweights_matrix[j][weight_index] # TODO forse devo dividerlo per len training set


    def compute_total_error(self, training_set, mse = True):
        ''' if mse is True it computes the Mean Square Error'''
        total_error = 0
        for t in range(len(training_set)):
            training_inputs, training_outputs = training_set[t]
            self.feed_forward(training_inputs)
            for o in range(len(training_outputs)):
                total_error += self.output_layer.neurons[o].mean_square_error(training_outputs[o])
        
        return total_error/len(training_set) if mse else total_error
    

    def __str__(self):
        nnd = "NeuraNetwork - inputs " + str(self.inputs_number) + " - learning rate " + str(self.learning_rate)
        nnd += "\nHidden " + str(self.hidden_layer)
        nnd += "\nOutput " + str(self.output_layer)
        return(nnd)
       