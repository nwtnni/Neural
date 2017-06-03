import numpy as np
import numpy.random as r
import math

# A simple neural network that using the logistic function.
class NeuralNetwork:

    # Layers is an array where layers[i] contains the number
    # of nodes in layer i.
    #
    # Rate is the learning rate for stochastic gradient descent.
    #
    def __init__(self, layers, rate):

        self.layers = layers
        self.weights = []
        self.rate = rate

        # Initialize weights
        for i in range(0, len(self.layers) - 1):
            self.weights.append(r.rand(self.layers[i + 1], self.layers[i]) - 0.5)

        print self.weights[0]

    def train(self):
        pass

    def query(self, input):

        values = input

        for layer in range(len(self.layers) - 1):
            values = np.dot(self.weights[layer], values)
            values = [logistic(x) for x in values]
        return values

def logistic(x):
    return 1.0 / (1 + math.exp(-x))

if __name__ == "__main__":
    r.seed(0)
    network = NeuralNetwork([3, 2], 1)
    print network.query(np.array([1, 1, 1]))
