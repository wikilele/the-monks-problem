import unittest
from NeuralNetwork.NeuronLayer import NeuronLayer


class MockWeightsService():    
    def get_bias(self):
        return 0.1
    
    def get_weights(self, weights_number):
        weights = []
        for i in range(weights_number): 
            weights.append(0.1)
        return weights

def test_feed_forward():
    mws = MockWeightsService()
    nl = NeuronLayer(4,3,mws)
    inputs = [4,3,2]
    outputs = nl.feed_forward(inputs)

    assert outputs[0] == outputs[1] and outputs[1] == outputs[2]

