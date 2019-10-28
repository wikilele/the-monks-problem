
import random

from DataSet import DataSet
from PlotService import PlotService
import first_neural_network.NeuralNetwork.NeuralNetwork as NN
import first_neural_network.NeuralNetwork.WeightsService as WS



# TODO
# sistemare il main
# prendere gli hyperparam dal json
# ROC curve?


def main():
    # read test file
    ps = PlotService()
    trs = DataSet('data/monks-1.train')
    
    # plotting the training data distribution
    ps.plot_distribution(trs.get_distribution())

    # nn
    ws = WS.WeightsService(-0.2,0.2)
    nn = NN.NeuralNetwork(1,17,3,1,ws)

    epoch = 320
    training_set = trs.get_set()
    plot_y = []   

    for i in range(epoch):
        random.shuffle(training_set)
        for pattern in training_set:
            training_input, training_output = pattern
            nn.train(training_input, training_output)
            
        error = nn.compute_total_error(training_set)
        plot_y.append(error)
        # print("epoch " + str(i) + " error " + str(error))

    ps.plot_error(range(epoch), plot_y)

    tss = DataSet('data/monks-1.test')
    accuracy = 0
    test_set = tss.get_set()

    for j in range(len(test_set)):
        output = nn.feed_forward(test_set[j][0])
        if output[0] >= 0.5 and test_set[j][1][0] == 1: # TruePositive
            accuracy +=1
        elif output[0] < 0.5 and test_set[j][1][0] == 0: # TrueNegative
            accuracy +=1
    
    print ("ACCURACY " + str(accuracy/len(test_set)*100) + "%") 

if __name__ == '__main__':
    main()