#
# test_and_or.py: creates various tic-tac-toe configurations 
#   for testing purposes
#
# Author: Derek Riley, 2020
#

from nn import NeuralNetwork
import numpy as np


def create_or_nn_data():
    # input training data set for OR
    x = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    # expected outputs corresponding to given inputs
    y = np.array([[0],
                  [1],
                  [1],
                  [1]])
    return x, y


def create_and_nn_data():
    # input training data set for AND
    x = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    # expected outputs corresponding to given inputs
    y = np.array([[0],
                  [0],
                  [0],
                  [1]])
    return x, y


def load_tictactoe_csv(filepath):
    boards = []
    labels = []
    with open(filepath) as fl:
        for ln in fl:
            cols = ln.strip().split(",")
            board = [0 if s == "o" else 1 for s in cols[:-1]]
            label = 0 if cols[-1] == "Cwin" else 1
            labels.append([label])
            boards.append(board)
    x = np.array(boards)
    y = np.array(labels)
    return x, y

def create_training_split(x, y, num_hidden_neurons, lr, epochs):
    shuffled = np.hstack((x, y))
    np.random.shuffle(shuffled)
    h_index = int(shuffled.shape[0] * .8)
    x_training_split = shuffled[:h_index, :-1]
    y_training_split = shuffled[:h_index, -1:]
    x_test_split = shuffled[h_index:, :-1]
    y_test_split = shuffled[h_index:, -1:]
    nn = NeuralNetwork(x_training_split, y_training_split, num_hidden_neurons)
    nn.train(epochs)
    print("Training accuracy, precision:", nn.accuracy_precision())
    nn.test_feedforward(x_test_split, y_test_split)
    print("Testing accuracy, precision:", nn.accuracy_precision())

def test_and_nn_1():
    x, y = create_and_nn_data()
    nn = NeuralNetwork(x, y, 4, 1)
    nn.train(150)
    print("test 4 loss:", nn.loss())
    # print("test 4 comparison:")
    # print("num hidden neurons:", 4)
    # print("num epochs:", 150)
    # print("learning rate:", 1)
    # create_training_split(x, y, 4, 1, 150)
    # print("100/0 split accuracy, precision:", nn.accuracy_precision())
    # print("-------")
    assert nn.loss() < .04




def test_and_nn_2():
    x, y = create_and_nn_data()
    nn = NeuralNetwork(x, y, 4, 2)
    nn.train(150)
    print("test 5 loss:", nn.loss())
    # print("test 5 comparison:")
    # print("num hidden neurons:", 4)
    # print("num epochs:", 150)
    # print("learning rate:", 2)
    # create_training_split(x, y, 4, 2, 150)
    # print("100/0 split accuracy, precision:", nn.accuracy_precision())
    # print("-------")
    assert nn.loss() < .01


def test_and_nn_3():
    x, y = create_and_nn_data()
    nn = NeuralNetwork(x, y, 3, 1)
    nn.train(1500)
    print("test 6 loss:", nn.loss())
    # print("test 6 comparison:")
    # print("num hidden neurons:", 3)
    # print("num epochs:", 1500)
    # print("learning rate:", 1)
    # create_training_split(x, y, 3, 1, 1500)
    # print("100/0 split accuracy, precision:", nn.accuracy_precision())
    # print("-------")
    assert nn.loss() < .002


def test_or_nn_1():
    x, y = create_or_nn_data()
    nn = NeuralNetwork(x, y, 4, 1)
    nn.train(150)
    print("test 1 loss:", nn.loss())
    # print("test 1 comparison:")
    # print("num hidden neurons:", 4)
    # print("num epochs:", 150)
    # print("learning rate:", 1)
    # create_training_split(x, y, 4, 1, 150)
    # print("100/0 split accuracy, precision:", nn.accuracy_precision())
    # print("-------")
    assert nn.loss() < .3


def test_or_nn_2():
    x, y = create_or_nn_data()
    nn = NeuralNetwork(x, y, 10, 1)
    nn.train(1000)
    print("test 2 loss:", nn.loss())
    # print("test 2 comparison:")
    # print("num hidden neurons:", 10)
    # print("num epochs:", 1000)
    # print("learning rate:", 1)
    # create_training_split(x, y, 10, 1, 1000)
    # print("100/0 split accuracy, precision:", nn.accuracy_precision())
    # print("-------")
    assert nn.loss() < .002


def test_or_nn_3():
    x, y = create_or_nn_data()
    nn = NeuralNetwork(x, y, 1, 2)
    nn.train(1500)
    print("test 3 loss:", nn.loss())
    # print("test 3 comparison:")
    # print("num hidden neurons:", 1)
    # print("num epochs:", 1500)
    # print("learning rate:", 1)
    # create_training_split(x, y, 1, 2, 1500)
    # print("100/0 split accuracy, precision:", nn.accuracy_precision())
    # print("-------")
    assert nn.loss() < .0009


def test_nn_1():
    x, y = load_tictactoe_csv("tic-tac-toe.csv")
    nn = NeuralNetwork(x, y, 4, .1)
    nn.train(1000)
    print("test 7 loss:", nn.loss())
    # print("test 7 comparison:")
    # print("num hidden neurons:", 4)
    # print("num epochs:", 1000)
    # print("learning rate:", .1)
    # create_training_split(x, y, 4, .1, 1000)
    # print("100/0 split accuracy, precision:", nn.accuracy_precision())
    # print("-------")
    assert nn.loss() < .06


def test_nn_2():
    x, y = load_tictactoe_csv("tic-tac-toe.csv")
    nn = NeuralNetwork(x, y, 10, .1)
    nn.train(10000)
    print("test 8 loss:", nn.loss())
    # print("test 8 comparison:")
    # print("num hidden neurons:", 10)
    # print("num epochs:", 10000)
    # print("learning rate:", .1)
    # create_training_split(x, y, 10, .1, 10000)
    # print("100/0 split accuracy, precision:", nn.accuracy_precision())
    # print("-------")
    assert nn.loss() < .0025


def test_nn_3():
    x, y = load_tictactoe_csv("tic-tac-toeFull.csv")
    nn = NeuralNetwork(x, y, 10, .04)
    nn.train(10000)
    # print("test 9 loss:", nn.loss())
    print("test 9 comparison:")
    print("num hidden neurons:", 10)
    print("num epochs:", 10000)
    print("learning rate:", .04)
    create_training_split(x, y, 10, .04, 10000)
    print("100/0 split accuracy, precision:", nn.accuracy_precision())
    print("-------")
    assert nn.loss() < .1


def test_nn_4():
    x, y = load_tictactoe_csv("tic-tac-toeFull.csv")
    nn = NeuralNetwork(x, y, 10, .03)
    nn.train(100000)
    # print("test 10 loss:", nn.loss())
    print("test 10 comparison:")
    print("num hidden neurons:", 10)
    print("num epochs:", 100000)
    print("learning rate:", .03)
    create_training_split(x, y, 10, .03, 100000)
    print("100/0 split accuracy, precision:", nn.accuracy_precision())
    print("-------")
    assert nn.loss() < .001


def test_nn_5():
    x, y = load_tictactoe_csv("tic-tac-toeFull.csv")
    nn = NeuralNetwork(x, y, 20, .01)
    nn.train(100000)
    # print("test 11 loss:", nn.loss())
    print("test 11 comparison:")
    print("num hidden neurons:", 20)
    print("num epochs:", 100000)
    print("learning rate:", .01)
    create_training_split(x, y, 20, .01, 100000)
    print("100/0 split accuracy, precision:", nn.accuracy_precision())
    print("-------")
    assert nn.loss() < .01

def test_nn_6():
    x, y = load_tictactoe_csv("tic-tac-toeFull.csv")
    nn = NeuralNetwork(x, y, 20, .002)
    nn.train(100000)
    # print("test 12 loss:", nn.loss())
    print("test 12 comparison:")
    print("num hidden neurons:", 20)
    print("num epochs:", 100000)
    print("learning rate:", .002)
    create_training_split(x, y, 30, .02, 100000)
    print("100/0 split accuracy, precision:", nn.accuracy_precision())
    print("-------")
    assert nn.loss() < .5


def run_all() -> None:
    """Runs all test cases"""
    test_or_nn_1()
    test_or_nn_2()
    test_or_nn_3()
    test_and_nn_1()
    test_and_nn_2()
    test_and_nn_3()
    test_nn_1()
    test_nn_2()
    test_nn_3()
    test_nn_4()
    test_nn_5()
    test_nn_6()
    print("All tests pass.")


def main() -> int:
    """Main test program which prompts user for tests to run and displays any
    result.
    """
    n = int(input("Enter test number (1-11; 0 = run all): "))
    if n == 0:
        run_all()
        return 0
    elif n == 1:
        result = test_or_nn_1()
    elif n == 2:
        result = test_or_nn_2()
    elif n == 3:
        result = test_or_nn_3()
    elif n == 4:
        result = test_and_nn_1()
    elif n == 5:
        result = test_and_nn_2()
    elif n == 6:
        result = test_and_nn_3()
    elif n == 7:
        result = test_nn_1()
    elif n == 8:
        result = test_nn_2()
    elif n == 9:
        result = test_nn_3()
    elif n == 10:
        result = test_nn_4()
    elif n == 11:
        result = test_nn_5()
    elif n == 12:
        result = test_nn_6()
    else:
        print("Error: unrecognized test number " + str(n))
    print("Test passes with result " + str(result))
    return 0


if __name__ == "__main__":
    exit(main())
