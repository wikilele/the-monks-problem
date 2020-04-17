import random
import math

class WeightsService():
    def __init__(self,lower_bound, upper_bound):
        ''' stores the weights upper and lower ranges '''
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
    
    def get_random_weight(self):
        ''' returns a random value in the range '''
        return random.uniform(self.lower_bound,self.upper_bound)

    def get_bias(self):
        return self.get_random_weight()
    
    def get_weights(self, weights_number):
        weights = []
        for i in range(weights_number):
            weights.append(self.get_random_weight())
        return weights

class GlorotBenjoWeightsService():
    ''' Glorot and Benjo ways of initializing weights'''

    def get_random_weight(self, fanin):
        ''' returns a random value in the range '''
        upper_bound = 1 / math.sqrt(fanin)
        lower_bound = - upper_bound
        return random.uniform(lower_bound,upper_bound)

    def get_bias(self):
        return 0
    
    def get_weights(self, weights_number):
        weights = []
        for i in range(weights_number):
            weights.append(self.get_random_weight(weights_number))
        return weights

        