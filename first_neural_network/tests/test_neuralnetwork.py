import unittest
from neuralnetwork import NeuralNetwork as myNN
from tests.pulled_neuralnetwork import NeuralNetwork as pulledNN


class MockWeightsService():    
    def get_bias(self):
        return 0.1
    
    def get_weights(self, weights_number):
        weights = []
        for i in range( weights_number): 
            weights.append(0.1 )
        return weights

def test_train1():
    mws = MockWeightsService()
    learning_rate = 0.5
    inputs_number = 3
    hidden_number = 3
    outputs_number = 1
    my_nn = myNN(learning_rate,inputs_number,hidden_number,outputs_number,mws)
    weights = mws.get_weights(3)
    bias = mws.get_bias()
    pulled_nn = pulledNN(inputs_number,hidden_number,outputs_number,weights,bias,weights,bias)

    inputs = [8,4,6]
    outputs = [1]
    training_set = [(inputs,outputs)]
    my_nn.online_training(training_set)
    pulled_nn.train(inputs, outputs)

    for i in range(inputs_number):
        for j,w in enumerate(my_nn.hidden_layer.neurons[i].weights):
            assert w == pulled_nn.hidden_layer.neurons[i].weights[j]
        
    for i in range(outputs_number ):
        for j,w in enumerate(my_nn.output_layer.neurons[i].weights):
            assert w == pulled_nn.output_layer.neurons[i].weights[j]


def test_total_error():
    mws = MockWeightsService()
    learning_rate = 0.5
    inputs_number = 3
    hidden_number = 3
    outputs_number = 1
    my_nn = myNN(learning_rate,inputs_number,hidden_number,outputs_number,mws)
    weights = mws.get_weights(3)
    bias = mws.get_bias()
    pulled_nn = pulledNN(inputs_number,hidden_number,outputs_number,weights,bias,weights,bias)

    inputs = [8,4,6]
    outputs = [1]
    training_set = [(inputs,outputs)]
    
    for i in range(100):
        my_nn.online_training(training_set)
        pulled_nn.train(inputs, outputs)
    
    assert my_nn.compute_total_error(training_set) == pulled_nn.calculate_total_error(training_set)

def test_weights_iterator():
    mws = MockWeightsService()
    learning_rate = 0.5
    inputs_number = 3
    hidden_number = 3
    outputs_number = 1
    my_nn = myNN(learning_rate,inputs_number,hidden_number,outputs_number,mws)

    iterations = 0
    for n, nu, wi in my_nn.weights_iterator(my_nn.hidden_layer):
        iterations += 1

    assert iterations == inputs_number*hidden_number

def test_batch_training():
    # TODO
    training_set = [
        [[0, 0, 0], [0]],
        [[0, 1, 1], [1]],
        [[1, 0, 0], [1]],
        [[1, 1, 1], [0]]
    ]
    mws = MockWeightsService()
    learning_rate = 0.5
    inputs_number = 3
    hidden_number = 3
    outputs_number = 1
    my_nn = myNN(learning_rate,inputs_number,hidden_number,outputs_number,mws)

    my_nn.batch_training(training_set)