import random
import math

class NeuronUnit:
    def __init__(self, bias, weights):
        self.bias = bias
        self.weights = weights
        self.inputs = []
        self.output = None

    def compute_output(self, inputs):
        ''' the function sets the class members self.input and self.output '''
        self.inputs = inputs
        self.output = self.logistic_function(self.net(inputs))
        return self.output

    def net(self, inputs):
        total = 0
        for i in range(len(inputs)):  
            total += inputs[i] * self.weights[i]
        return total + self.bias

    def logistic_function(self, net):
        ''' Apply the logistic function to squash the output of the neuron '''
        return 1 / (1 + math.exp(-net))

    def mean_square_error(self, target_output):
        ''' the error for each neuron is calculated by the Mean Square Error method (care for 0.5) '''
        return 0.5 * (target_output - self.output) ** 2
    
    def dE_dout(self,target_output):
        ''' derivative of error w.r.t. output. The constant 2 is simplified with 1/2 (see notes and self.mean_square_error) '''
        return - (target_output- self.output)

    def dout_dnet(self):
        ''' first derivative of logistic function '''
        return self.output * (1 - self.output)

    def dnet_dwj(self,j):
        ''' just input at position j'''
        return self.inputs[j]

    # check the notes
    def dE_dwj(self,target_output,j):
        ''' by chain rule ''' 
        return self.dE_dout(target_output) * self.dout_dnet() * self.dnet_dwj(j)

    def delta(self,target_output):
        ''' this is correct only if the node is an output one '''
        return self.dE_dout(target_output) * self.dout_dnet()
    
    def __str__(self):
       return "Neuron Unit - bias " + "%.3f" % self.bias + " - weights " +  str([ float("%.3f" % w) for w in  self.weights])