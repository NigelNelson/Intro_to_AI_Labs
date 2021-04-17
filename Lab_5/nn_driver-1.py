from typing import *
import numpy as np
from nn import NeuralNetwork

def create_or_nn_data():
    # input training data set for OR
    x = np.array([[0,0],
                [0,1],
                [1,0],
                [1,1]])
    # expected outputs corresponding to given inputs
    y = np.array([[0],
                [1],
                [1],
                [1]])
    return x,y

def create_and_nn_data():
    # input training data set for AND
    x = np.array([[0,0],
                [0,1],
                [1,0],
                [1,1]])
    # expected outputs corresponding to given inputs
    y = np.array([[0],
                [0],
                [0],
                [1]])
    return x,y
    
def load_tictactoe_csv(filepath):
    # FIXME week 6
    # good
    data = np.genfromtxt(filepath, delimiter=',', dtype=np.str)
    y = data[:, 9:] == "Cwin"
    return data[:, :-1] == 'x', y.astype(int)

def test_or_nn(verbose=0):
    x,y = create_or_nn_data()
    nn = NeuralNetwork(x, y, 4, 1)
    nn.feedforward()
    if verbose > 0:
        print("OR 1 " + str(nn.loss()))
        print("NN output " + str(nn._output))
        print(nn.accuracy_precision())
    assert nn.loss() < .04

def test_ttt_nn(verbose=0):
    x, y = load_tictactoe_csv("tic-tac-toe-1.csv")
    nn = NeuralNetwork(x, y, 4, .1)
    nn.load_4_layer_ttt_network()
    nn.feedforward()
    if verbose > 0:
        print("NN 1 " + str(nn.loss()))
        print("NN output " + str(nn._output))
        print(nn.accuracy_precision())
    assert nn.loss() < .02

def run_all(verbose=0) -> None:
    """Runs all test cases"""
    test_or_nn(verbose)
    test_ttt_nn(verbose)
    print("All tests pass.")


def main() -> int:
    """Main test program which prompts user for tests to run and displays any
    result.
    """
    verbose = int(input("Enter 0-2 for verbosity (0 is quiet, 2 is everything):"))
    n = int(input("Enter test number (1-2; 0 = run all): "))
    verbose = 2;
    if n == 0:
        run_all(verbose)
        return 0
    elif n == 1:
        result = test_or_nn(verbose)
    elif n == 2:
        result = test_ttt_nn(verbose)
    else:
        print("Error: unrecognized test number " + str(n))
    print("Test passes with result " + str(result))
    return 0


if __name__ == "__main__":
    main()