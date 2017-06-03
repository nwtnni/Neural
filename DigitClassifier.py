from NeuralNetwork import NeuralNetwork
import numpy as np
from numpy.random import rand

def extract(filename):

    examples = []
    targets = []

    with open(filename) as f:
        for line in f:

            line = line.split(",")
            pixels = np.asfarray(line[1:])
            pixels = (pixels / 255.0 * 0.99) + .01
            examples.append(pixels[:,None])

            target = np.zeros(10) + 0.01
            target[int(line[0])] = 0.99
            targets.append(target[:,None])

    return (examples, targets)

if __name__ == "__main__":

    network = NeuralNetwork([784, 100, 10], 0.3)

    (examples, targets) = extract("./Small/mnist_train_100.csv")

    network.train(examples, targets, 1)

    (examples, targets) = extract("./Small/mnist_test_10.csv")

    for i in range(len(targets)):

        result = np.argmax(network.query(examples[i]))
        print "Expected ", np.argmax(targets[i]), ", got ", result
