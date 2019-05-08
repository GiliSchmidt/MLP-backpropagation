import numpy as np

debug = False


class Layer:
    def __init__(self, input_count, neurons_count, bias=0.5, weights=None):
        self.input_count = input_count
        self.neurons_count = neurons_count
        self.weights = weights if weights is not None else np.random.rand(self.input_count, self.neurons_count)
        self.bias = bias
        # will be updated to array after first calc_output
        self.output = 0.0
        self.error = 0.0
        self.delta = 0.0

        if debug:
            print("New layer: \nneurons: ", neurons_count, "\ninputs: ", input_count, '\nW count: ', len(self.weights),
                  '\nBias:', self.bias)
            print('---')


class MiddleLayer(Layer):
    def __init__(self, input_count, neurons_count, bias, weights=None):
        Layer.__init__(self, input_count, neurons_count, bias, weights)

    def calc_output(self, X):
        res = np.dot(X, self.weights) + self.bias
        self.output = self.sigmoid(res)

        if debug:
            print("output: ", self.output)
        return self.output

    # activation
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivate(self, z):
        return z * (1 - z)

    def calc_error(self, expected):
        self.error = np.dot(expected.weights, expected.delta)
        self.delta = self.error * self.sigmoid_derivate(self.output)

        if (debug):
            print("error: ", self.error, " delta: ", self.delta)

    def update_weights(self, values, learning_rate):
        if (debug):
            old = self.weights.copy()

        new = (np.matrix(values).T * self.delta) * learning_rate
        self.weights += new

        if (debug):
            print("updating weights: \nold: ", old, " \nnew:", self.weights)


class OutputLayer(MiddleLayer):
    def __init__(self, input_count, neurons_count, bias, weights=None):
        MiddleLayer.__init__(self, input_count, neurons_count, bias, weights)

    def calc_error(self, expected):
        self.error = expected - self.output
        self.delta = self.error * self.sigmoid_derivate(self.output)

        if (debug):
            print("error: ", self.error, " delta: ", self.delta)
