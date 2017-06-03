import numpy as np
import numpy.random as r
import math

# Returns a column vector
def logistic(vector):
    return np.array([[(1.0 / (1 + math.exp(-x))) for x in vector]]).T

# A simple neural network that uses the logistic function.
class NeuralNetwork:

    # Layers is an array where layers[i] contains the number
    # of nodes in layer i.
    #
    # Rate is the learning rate for stochastic gradient descent.
    #
    def __init__(self, layers, rate, fn=logistic):

        self.layers = layers
        self.weights = []
        self.rate = rate
        self.fn = fn

        # Initialize weights
        for i in range(0, len(self.layers) - 1):
            self.weights.append(r.rand(self.layers[i + 1], self.layers[i]) - 0.5)

    # Train using back-propagation and stochastic gradient descent
    def train(self, examples, targets, epochs):

        for epoch in range(epochs):

            # Iterate through all training examples
            for i in range(len(examples)):


                # Forward phase
                out = [examples[i]]

                for layer in range(len(self.layers) - 1):
                    out.append(self.fn(np.dot(self.weights[layer], out[-1])))

                error = targets[i] - out[-1]

                if (i % 100 == 0):
                    print "Error is ", np.linalg.norm(error), " on iteration ", i, " for input ", np.argmax(targets[i])

                # Backward phase
                for i in range(len(self.layers) - 1, 0, -1):

                    delta = self.rate * np.dot(error * out[i] * (1 - out[i]), out[i - 1].T)
                    error = np.dot(self.weights[i - 1].T, error)
                    self.weights[i - 1] += delta

    # Feed-forward to get the predicted result
    def query(self, input):

        values = input

        for layer in range(len(self.layers) - 1):
            values = self.fn(np.dot(self.weights[layer], values))

        return values
