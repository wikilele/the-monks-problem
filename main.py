
from DataSet import DataSet
import first_neural_network.NeuralNetwork.NeuralNetwork as NN
import first_neural_network.NeuralNetwork.WeightsService as WS
import random

def main():
    # read test file
    trs = DataSet('data/monks-1.train')
    #print( trs.get_set() )

    # nn
    ws = WS.WeightsService(-0.2,0.2)
    nn = NN.NeuralNetwork(2,17,3,1,ws)

    epoch = 320
    training_set = trs.get_set()

    for i in range(epoch):
        random.shuffle(training_set)
        for pattern in training_set:
            training_input, training_output = pattern
            nn.train(training_input, training_output)
            
        error = nn.compute_total_error(training_set)
        print("epoch " + str(i) + " error " + str(error))

    tss = DataSet('data/monks-1.test')
    accuracy = 0
    test_set = tss.get_set()

    for j in range(len(test_set)):
        output = nn.feed_forward(test_set[j][0])
        if output[0] >= 0.5 and test_set[j][1][0] == 1: # TruePositive
            accuracy +=1
        elif output[0] < 0.5 and test_set[j][1][0] == 0: # TrueNegative
            accuracy +=1
    
    print (accuracy/len(test_set)) 

if __name__ == '__main__':
    main()