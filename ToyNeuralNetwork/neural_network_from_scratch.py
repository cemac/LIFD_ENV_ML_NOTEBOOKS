"""NeuralNetwork Example
This is a simple example  of a 2 layer neural network written in python based
off of James Loy (Georgia Tech) work
"""

import numpy as np


def sigmoid(x):
    """sigmoid
    activation function
    """
    return 1.0/(1 + np.exp(-x))


def sigmoid_derivative(x):
    """sigmoid_derivative
    the derivative of sigma(x)
    d sig(x)/ dx!
    """
    return x * (1.0 - x)


class NeuralNetwork:
    """Neural Network
    Description:

    Attributes:
        x: input layer
        y: output layer
        weights: W1 and W1
    """
    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], 4)
        self.weights2 = np.random.rand(4, 1)
        self.y = y
        self.output = np.zeros(y.shape)

    def feedforward(self):
        """A freeford function
            z = sigma ( W1 x+ b1)
            y = sigma (W2 z + b2)
        Attributes:
            b = 0 (biases set to zero for ease)
            layer1 = z
        """
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        """
        application of the chain rule to find derivative of the loss function
        with respect to weights2 and weights1
        d Loss(y,yp)/ dw = dL(y,yp)/dyp * dy / dz * dz/ dW
        Attributes:
            b = 0 for ease
            sigmoid_derivative = derivative of sigma
        """
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) *
                            sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) *
                            sigmoid_derivative(self.output), self.weights2.T) *
                            sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2


if __name__ == "__main__":
    X = np.array([[0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])
    y = np.array([[0], [1], [1], [0]])
    nn = NeuralNetwork(X, y)

    for i in range(1500):
        nn.feedforward()
        nn.backprop()
    print(nn.output)
