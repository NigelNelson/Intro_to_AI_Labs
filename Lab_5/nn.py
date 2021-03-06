#
# nn.py: Basic Neural Network implementation stub.  
# You will fill out the stubs below using numpy as much as possible.  
# This class serves as a base for you to build on for the labs.
#
# Lab 6: Neural Networks
# Date: 4/15/21
# Course: CS2400
# Author(s): Derek Riley, Nigel Nelson
#

import numpy as np


def sigmoid(x):
    """This is the sigmoid activation function."""
    return 1.0 / (1.0 + np.exp(-1.0 * x))


def sigmoid_derivative(x):
    """This is the derivative of the sigmoid function."""
    return sigmoid(x) * (1.0 - sigmoid(x))


class NeuralNetwork:
    """Represents a basic fully connected single-layer neural network.  

    Attributes:
        input (2D numpy array): input features, one row for each sample, 
            and one column for each feature
        weights2 (numpy array): connection weights between the input
            and hidden layer
        weights3 (numpy array): connection weights between the hidden
            layer and output neuron
        y (numpy array): expected outputs of the network, one row for each 
            sample, and one column for each output variable
        output (numpy array): stores the current output of the network 
            after a feedforward pass
        learning_rate (float): scales the derivative influence in backprop
    """

    def __init__(self, x, y, num_hidden_neurons=4, lr=1):
        """Setup a Neural Network with a single hidden layer.  This method
        requires two vectors of x and y values as the input and output data.
        """
        # FIXME week 7
        self._a_1 = x
        self._weights_2 = np.array([[3.07153357, 2.01940447, -2.14695621,
                                     2.62044111],
                                    [2.83203743, 2.15003442, -2.16855273,
                                     2.77165525]])
        self._weights_3 = np.array([[3.8124126],
                                    [1.92454886],
                                    [-5.20663292],
                                    [3.21598943]])
        self._biases_2 = np.array([-1.26285168, -0.72768134, 0.89760201,
                                   -1.10572122])
        self._biases_3 = np.array([-2.1110666])
        self._y = y
        self._output = np.zeros(self._y.shape)
        self._learning_rate = lr

    def load_4_layer_ttt_network(self):
        self._weights_2 = np.array([[-3.12064667, -0.62044264, -3.18868069,
                                     -1.06183619],
                                    [-2.75995675, -0.3063746, -3.24168826,
                                     -0.7056788],
                                    [0.35471861, -1.40337629, 0.3368032,
                                     1.96311844],
                                    [0.31900681, -0.98534514, 0.36569296,
                                     1.7516015],
                                    [1.18823403, -0.88661356, 1.42729163,
                                     2.3146592],
                                    [2.24817726, -0.73170809, 2.42017968,
                                     3.13494424],
                                    [2.43338048, -1.12167492, 2.78634464,
                                     3.30680788],
                                    [1.57132788, -1.4313579, 1.66389342,
                                     2.45366816],
                                    [1.4126572, -1.38204671, 1.45066697,
                                     2.78777504]])
        self._weights_3 = np.array([[6.10550764],
                                    [2.6696074],
                                    [6.58122877],
                                    [-5.46573692]])
        self._biases_2 = np.array([-0.00142707, -0.08451622, -0.00777166,
                                   0.07153606])
        self._biases_3 = np.array([0.03276832])

    def inference(self):
        """
        Uses sigmoid activation function applied to the input * weights + bias
        """
        hidden_layer = sigmoid(np.dot(self._a_1, self._weights_2) +
                               self._biases_2)
        return sigmoid(np.dot(hidden_layer, self._weights_3) +
                       self._biases_3)

    def feedforward(self):
        """
        This is used in the training process to calculate and save the
        outputs for backpropogation calculations.
        """
        self._output = self.inference()

    def get_binary_output(self):
        """
        Uses a threshold of .7 to map output to a boolean matrix, which is then
        converted to an int matrix.
        """
        return (self._output > .7).astype(int)

    def backprop(self):
        """
        Update model weights based on the error between the most recent
        predictions (feedforward) and the training values.
        """
        # FIXME week 7

    def train(self, epochs=100, verbose=0):
        """This method trains the network for the given number of epochs.
        It doesn't return anything, instead it just updates the state of
        the network variables.
        """
        for i in range(epochs):
            self.feedforward()
            self.backprop()
            if verbose > 1:
                print(self.loss())

    def loss(self):
        """ Calculate the MSE error for the set of training data."""
        return np.mean(np.square(self.get_binary_output() - self._y))

    def accuracy_precision(self):
        """accuracy = Total correct prediction / total num predication.
        Precision = True positives / all positives"""
        accuracy = (self.get_binary_output() == self._y).sum() / self._y.size
        precision = ((self.get_binary_output() +
                      self._y) == 2).sum() / self.get_binary_output().sum()
        return accuracy, precision
