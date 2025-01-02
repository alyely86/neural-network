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
        
    def train(self,inputs_list,targets_list):
        #convert inputs list to 2d array
        inputs = numpy.array(inputs_list,ndmin=2).T
        targets = numpy.array(targets_list,ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih,inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)


        #clculate signals into final output layer 
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer 
        final_outputs = self.activation_function(final_inputs)
        
        pass


    def query(self,input_list):
        # convert inputs list to 2d array
        inputs = numpy.array(input_list,ndmin=2).T

        #calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # caculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who,hidden_outputs)
        # calculate the signals emerging from final outputs layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


input_nodes = 3
hidden_nodes = 3
output_nodes = 3

learning_rate = 0.3

n = neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)
print(n.query([1.0,0.5,-1.5]))
