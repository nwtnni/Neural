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

    def train(self, examples, targets, epochs):

        for epoch in range(epochs):

            # Iterate through all training examples
            for i in range(len(examples)):

                # Forward phase
                out = [examples[i]]
                for layer in range(len(self.layers) - 1):
                    out.append(logistic(np.dot(self.weights[layer], out[-1])))

                error = targets[i] - out[-1]

                # Backward phase
                for i in range(len(self.layers) - 1, 0, -1):

                    delta = self.rate + np.dot(error * logistic(out[i]) * 1 - logistic(out[i]), out[i - 1].T)
                    error *= self.weights[i - 1].T
                    self.weights[i - 1] += delta

    def query(self, input):

        values = input

        for layer in range(len(self.layers) - 1):
            values = logistic(np.dot(self.weights[layer], values))

        return values

def logistic(vector):
    return [(1.0 / (1 + math.exp(-x))) for x in vector]

if __name__ == "__main__":
    r.seed(0)
    network = NeuralNetwork([3, 2, 7, 100, 200], 0.1)
    print network.query(np.array([3, 3, 3]))
