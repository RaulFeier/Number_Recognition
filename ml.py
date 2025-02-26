import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# test something
class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.layers = sizes
        self.biases = [np.random.randn(1, y) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

n = Network([3, 3, 2])
