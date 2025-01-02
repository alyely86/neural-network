import numpy
import scipy.special as sp

class neuralNetwork:

    def __init__(self,inputnodes,hiddennodes,outputnodes,   learningrate):
        # Set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        #link weight matrces, wih and who
        #weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        #w11 w21
        #w12 w22 etc
        self.wih = (numpy.random.rand(self.hnodes,self.inodes)-0.5)
        self.who = (numpy.random.rand(self.onodes,self.hnodes)-0.5)

        #learning rate 
        self.lr = learningrate

        #activaion function is the sigmoid function
        self.activation_function = lambda x: sp.expit(x)
        
    def train():
        pass

    def query():
        pass


input_nodes = 3
hidden_nodes = 3
output_nodes = 3

learning_rate = 0.5

n = neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)
