from NeuralNetwork import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    layers = [2, 4, 1]
    rate = 0.2

    network = NeuralNetwork(layers, rate)

    train_data = [np.array([0.01, .01]), np.array([0.01,0.99]), np.array([0.99, 0.01]), np.array([0.99, 0.99])]
    train_labels = [np.array([0.01]), np.array([0.99]), np.array([0.99]), np.array([0.01])]

    print "Network created:"
    print "\tLayers:       ", layers
    print "\tRate:         ", rate, "\n"

    for i, data in enumerate(train_data):

        print "Before training:"
        print "\tQuerying with: ", train_data[i]
        print "\tExpected:      ", train_labels[i]

        train_data[i] = train_data[i][:,None]
        train_labels[i] = train_labels[i][:,None]

        print "\tObtained:      ", max(network.query(train_data[i]))

    print "\nTraining...\n"
    network.train(train_data, train_labels, 10000) 

    for i, data in enumerate(train_data):

        print "After training:"
        print "\tQuerying with: ", train_data[i].T
        print "\tExpected:      ", train_labels[i].T
        print "\tObtained:      ", max(network.query(train_data[i]))
