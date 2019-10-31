
import random
import json

from DataSet import DataSet
from PlotService import PlotService
import first_neural_network.NeuralNetwork.NeuralNetwork as NN
import first_neural_network.NeuralNetwork.WeightsService as WS



# TODO
# ROC curve?
# 1 - batch training - needs a code review
# 3 - regularization
# cross validation
# 4 - momentum
# 2 - Glorot Benjo weight init - nnes to have a small test (there is a second way to do it) http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf


def main():
    # read test file
    ps = PlotService()
    trs = DataSet('data/monks-1.train')

    # plotting the training data distribution
    # ps.plot_distribution(trs.get_distribution())

    # getting hyperparameters
    with open('hyperparams1.json','r') as f:
        hyp= json.load(f)

 
    # nn
    ws = WS.WeightsService(hyp['weights_lowerbound'],hyp['weights_upperbound'])
    nn = NN.NeuralNetwork(hyp['learning_rate'],hyp['input_units'],hyp['hidden_units'],hyp['output_units'],ws)

    epoch = hyp['epochs']
    training_set = trs.get_set()
    plot_y = []   

    for i in range(epoch):
        nn.online_training(training_set)
        #nn.batch_training(training_set)
        error = nn.compute_total_error(training_set, mse=True) 
        plot_y.append(error)
        # print("epoch " + str(i) + " error " + str(error)) 

    tss = DataSet('data/monks-1.test')
    accuracy = 0
    test_set = tss.get_set()
    threshold = hyp['threshold']

    for j in range(len(test_set)):
        output = nn.feed_forward(test_set[j][0])
        if output[0] >= threshold and test_set[j][1][0] == 1: # TruePositive
            accuracy +=1
        elif output[0] < threshold and test_set[j][1][0] == 0: # TrueNegative
            accuracy +=1
    
    print ("ACCURACY " + str(accuracy/len(test_set)*100) + "%") 
    ps.plot_error(range(epoch), plot_y)

if __name__ == '__main__':
    main()