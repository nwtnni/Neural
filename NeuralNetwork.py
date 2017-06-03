import numpy as np
import numpy.random as r
import math

# Returns a column vector
def logistic(vector):
    return np.array([[(1.0 / (1 + math.exp(-x))) for x in vector]]).T

def invlog(vector):
    return np.array([[math.log(x / (1.0 - x)) for x in vector]]).T

# A simple neural network that uses the logistic function.
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

    # Train using back-propagation and stochastic gradient descent
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

                    delta = self.rate * np.dot(error * out[i] * (1 - out[i]), out[i - 1].T)
                    error = np.dot(self.weights[i - 1].T, error)
                    self.weights[i - 1] += delta

    # Feed-forward to get the predicted result
    def query(self, input):

        values = input

        for layer in range(len(self.layers) - 1):
            values = logistic(np.dot(self.weights[layer], values))

        return values

    # Feed-backward to get a representative input
    def backQuery(self, output):

        values = output

        for layer in range(len(self.layers) - 2, -1, -1):
            values = np.dot(self.weights[layer].T, values)
            values -= np.min(values)
            values /= np.max(values)
            values *= 0.98
            values += 0.01
            values = invlog(values)

        return values
