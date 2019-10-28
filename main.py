
from TrainingSet import TrainingSet
import first_neural_network.NeuralNetwork.NeuralNetwork as NN
import first_neural_network.NeuralNetwork.WeightsService as WS
import random

def main():
    # read test file
    trs = TrainingSet('data/monks-1.test')
    #print( trs.get_set() )

    # nn
    ws = WS.WeightsService(-0.7,0.7)
    nn = NN.NeuralNetwork(2,17,3,1,ws)

    epoch = 320
    training_set = trs.get_set()
    for i in range(epoch):
        training_inputs, training_outputs = random.choice(training_set)
        nn.train(training_inputs, training_outputs)
        error = nn.compute_total_error(training_set)
        print("epoch " + str(i) + " error " + str(error))

    
if __name__ == '__main__':
    main()