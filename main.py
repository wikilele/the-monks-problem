
from TrainingSet import TrainingSet
NN = __import__('first-neural-network.NeuralNetwork')

def main():
    # read test file
    ts = TrainingSet('data/monks-1.test')
    print( ts.get_set() )

    # nn
    ws = NN.WeightsService(-0.7,0.7,4)
    nn = NN.NeuralNetwork(0.5,3,3,1,ws)



if __name__ == '__main__':
    main()