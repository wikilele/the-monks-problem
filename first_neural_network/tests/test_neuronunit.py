import unittest
from neuralnetwork.neuronunit import NeuronUnit
from tests.pulled_neuralnetwork import Neuron

def test_compute_output1():
    ''' all weights setted to 0 '''
    nu = NeuronUnit(bias = 0, weights = [0,0,0,0])
    output = nu.compute_output([7,7,7,7])
    assert output == 0.5

def test_compute_output2():
    nu = NeuronUnit(bias = 0, weights = [0,1,0,0])
    output = nu.compute_output([0,1,0,0])
    assert output == 0.7310585786300049

def test_derivatives1():
    bias = 0.1
    weights = [0.01,-0.2,-0.001,0.3]
    inputs = [1,2,3,4]
    my_nu = NeuronUnit(bias,weights)
    pulled_nu = Neuron(bias)
    pulled_nu.weights = weights
    target_output = 1
    index_j = 2

    # forward pass is needed, else we do not have the output computed
    assert my_nu.compute_output(inputs) == pulled_nu.calculate_output(inputs)

    # testing delta
    assert my_nu.delta(target_output) == pulled_nu.calculate_pd_error_wrt_total_net_input(target_output)

    # this tests are quite trivial the main change is in the function name
    assert my_nu.dE_dout(target_output) == pulled_nu.calculate_pd_error_wrt_output(target_output)
    assert my_nu.dout_dnet() == pulled_nu.calculate_pd_total_net_input_wrt_input()
    assert my_nu.dnet_dwj(index_j) == pulled_nu.calculate_pd_total_net_input_wrt_weight(index_j)

    # testing the whole Error partial derivative
    temp = pulled_nu.calculate_pd_error_wrt_total_net_input(target_output) * pulled_nu.calculate_pd_total_net_input_wrt_weight(index_j)
    assert my_nu.dE_dwj(target_output,index_j) == temp


