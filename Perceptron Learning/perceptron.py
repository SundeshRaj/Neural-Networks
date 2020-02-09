# Raj, Sundesh
# 1001-633-297
# 2019-09-22
# Assignment-01-01

import numpy as np
import itertools

class Perceptron(object):
    def __init__(self, input_dimensions=2,number_of_classes=4,seed=None):
        """
        Initialize Perceptron model
        :param input_dimensions: The number of features of the input data, for example (height, weight) would be two features.
        :param number_of_classes: The number of classes.
        :param seed: Random number generator seed.
        """
        if seed != None:
            np.random.seed(seed)
        self.input_dimensions = input_dimensions
        self.number_of_classes=number_of_classes
        self._initialize_weights()
        
    def _initialize_weights(self):
        """
        Initialize the weights, initalize using random numbers.
        Note that number of neurons in the model is equal to the number of classes
        """
        
        self.weights = np.random.randn(self.number_of_classes,self.input_dimensions+1)
        

    def initialize_all_weights_to_zeros(self):
        """
        Initialize the weights, initalize using random numbers.
        """
            
        self.weights = np.zeros((self.number_of_classes,self.input_dimensions+1))      
       

    def predict(self, X):
        """
        Make a prediction on an array of inputs
        :param X: Array of input [input_dimensions,n_samples]. Note that the input X does not include a row of ones
        as the first row.
        :return: Array of model outputs [number_of_classes ,n_samples]
        """
        
        ones = X.shape[1]     
        X = np.vstack((np.ones(ones),X))
        predictedValue=np.dot(self.weights,X)
        
        predictedValue[predictedValue >= 0] = 1
        predictedValue[predictedValue < 0 ] = 0
        
        return predictedValue

    def print_weights(self):
        
        """
        This function prints the weight matrix (Bias is included in the weight matrix).
        """
        print(self.weights)
        
    def train(self, X, Y, num_epochs=10, alpha=0.001):
        """
        Given a batch of data, and the necessary hyperparameters,
        this function adjusts the self.weights using Perceptron learning rule.
        Training should be repeted num_epochs time.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_classes ,n_samples]
        :param num_epochs: Number of times training should be repeated over all input data
        :param alpha: Learning rate
        :return: None
        """
        
        ones = X.shape[1]
        X = np.vstack((np.ones(ones),X))#add a row of 1s to the input array to compensate for bias in the weight matrix

        for epoch in range(num_epochs):
            for i in range(X.shape[1]):
                p = X[:,i]
                target = Y[:,i]
                p = p.reshape(self.input_dimensions+1,1)
                target = target.reshape(self.number_of_classes,1)
                actual = np.dot(self.weights,p)
                actual[actual >= 0] = 1
                actual[actual < 0 ] = 0
                self.weights += alpha*(np.dot((target-actual),p.T))

    def calculate_percent_error(self,X, Y):
        """
        Given a batch of data this function calculates percent error.
        For each input sample, if the output is not hte same as the desired output, Y,
        then it is considered one error. Percent error is number_of_errors/ number_of_samples.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_classes ,n_samples]
        :return percent_error
        """
        
        one = X.shape[1]      
        X = np.vstack((np.ones(one),X))
        result=np.dot(self.weights,X)
        
        result[result >= 0] = 1
        result[result < 0 ] = 0
        result.astype(int)
        
        error = 0
        for i in range(X.shape[1]):
            #array compare actual vs target to calculate percent error
            #https://docs.scipy.org/doc/numpy/reference/generated/numpy.array_equal.html
            if np.array_equal(result[:,i],Y[:,i]) == False:
                error += 1
        return error/(X.shape[1])


if __name__ == "__main__":
    """
    This main program is a sample of how to run your program.
    You may modify this main program as you desire.
    """

    input_dimensions = 2
    number_of_classes = 2

    model = Perceptron(input_dimensions=input_dimensions, number_of_classes=number_of_classes, seed=1)
    X_train = np.array([[-1.43815556, 0.10089809, -1.25432937, 1.48410426],
                        [-1.81784194, 0.42935033, -1.2806198, 0.06527391]])
    #print("********predicted output********\n",model.predict(X_train))
    Y_train = np.array([[1, 0, 0, 1], [0, 1, 1, 0]])
    model.initialize_all_weights_to_zeros()
    #print("****** Model weights ******\n",model.weights)
    #print("****** Input samples ******\n",X_train)
    #print("****** Desired Output ******\n",Y_train)
    percent_error=[]
    for k in range (20):
        model.train(X_train, Y_train, num_epochs=1, alpha=0.0001)
        percent_error.append(model.calculate_percent_error(X_train,Y_train))
    #print("******  Percent Error ******\n",percent_error)
    #print("****** Print weights *******")
    #model.print_weights()