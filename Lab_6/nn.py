#
# nn.py: Neural Network implementation that utilizes
# numpy matrices and arrays to implement forward propagation
# and backpropagation.
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
        self._a_1 = x
        self._weights_2 = np.random.rand(x.shape[1], num_hidden_neurons)
        self._weights_3 = np.random.rand(num_hidden_neurons, 1)
        self._biases_2 = np.random.rand(num_hidden_neurons)
        self._biases_3 = np.random.rand(1)
        self._y = y
        self._output = np.zeros(self._y.shape)
        self._learning_rate = lr
        self._z_l2_values = np.zeros((x.shape[0], num_hidden_neurons))
        self._a_l2_values = np.zeros((x.shape[0], num_hidden_neurons))
        self._z_l3_values = np.zeros((x.shape[0], 1))

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
        self._z_l2_values = np.dot(self._a_1, self._weights_2) + self._biases_2
        self._a_l2_values = sigmoid(self._z_l2_values)
        self._z_l3_values = np.dot(self._a_l2_values,
                                   self._weights_3) + self._biases_3
        return sigmoid(self._z_l3_values)

    def feedforward(self):
        """
        This is used in the training process to calculate and save the
        outputs for backpropogation calculations.
        """
        self._output = self.inference()

    def test_feedforward(self, test_x, test_y):
        """
        This is used to pass validation data through a network that has been
        trained on a subset of a complete dataset.
        """
        self._a_1 = test_x
        self._y = test_y
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
        DL/DB only needs to be calculated for layers.
        dldw is matrix, dl/db is array
        store z and a values
        """
        # Calculating the delta error for the final layer
        # shape = (num input columns, 1)
        delta_error_l3 = (self._output - self._y) * sigmoid_derivative(
            self._z_l3_values)
        # Calculating the delta error for the hidden layer
        # shape = (num input columns, num nodes in hidden layer)
        delta_error_l2 = np.dot(delta_error_l3,
                                self._weights_3.T) * sigmoid_derivative(
            self._z_l2_values)
        # Final Layer delta bias error
        delta_bias_error_l3 = delta_error_l3.sum(axis=0)
        # Hidden Layer delta bias error
        delta_bias_error_l2 = delta_error_l2.sum(axis=0)
        # Final layer delta weight error
        # shape = (Num nodes in hidden layer, 1)
        delta_weight_error_l3 = np.dot(self._a_l2_values.T, delta_error_l3)
        # Hidden layer delta weight error
        # shape = (Num elements in a input row, Num nodes in hidden layer)
        delta_weight_error_l2 = np.dot(self._a_1.T, delta_error_l2)
        # Updating final layer weights and biases
        self._weights_3 += -1 * (delta_weight_error_l3 * self._learning_rate)
        self._biases_3 += -1 * (delta_bias_error_l3 * self._learning_rate)
        # Updating hidden layer weights and biases
        self._weights_2 += -1 * (delta_weight_error_l2 * self._learning_rate)
        self._biases_2 += -1 * (delta_bias_error_l2 * self._learning_rate)

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
        return np.mean(np.square(self._output - self._y))

    def accuracy_precision(self):
        """accuracy = Total correct prediction / total num predication.
        Precision = True positives / all positives"""
        accuracy = (self.get_binary_output() == self._y).sum() / self._y.size
        if self.get_binary_output().sum() == 0:
            precision = 0.0
        else:
            precision = ((self.get_binary_output() +
                          self._y) == 2).sum() / self.get_binary_output().sum()
        return accuracy, precision
