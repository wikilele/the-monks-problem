from .neuronunit import NeuronUnit

class NeuronLayer:
    def __init__(self, neurons_number, weights_number,  weights_service):
        self.outputs = []
        self.neurons = []
        for i in range(neurons_number):
            nu = NeuronUnit( weights_service.get_bias(),  weights_service.get_weights(weights_number))
            self.neurons.append(nu)

    def feed_forward(self, inputs):
        self.outputs = []
        for neuron in self.neurons:
            self.outputs.append(neuron.compute_output(inputs))
        return self.outputs

    def __str__(self):
        nld =  "NeuronLayer - with " + str(len(self.neurons)) + " neuron units"
        for nu in self.neurons:
            nld += "\n\t" + str(nu)
        return nld