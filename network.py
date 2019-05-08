from layers import *
import numpy as np
import os
import glob

debug = False


class Network:
    """
    input_size - size of layer Input
    hidden_size[] - array containing the size of each hidden layer
    output_size - size of layer Output
    hidden_bias[] -array of bias for each hidden layer
    output_bias - output bias
    """

    def __init__(self, input_size, hidden_size, output_size, hidden_bias=None, output_bias=0.05):
        hidden_bias = hidden_bias if hidden_bias is not None else [0.05 for i in hidden_size]

        self.layers = []
        self.layers.append(MiddleLayer(input_size, hidden_size[0], bias=hidden_bias[0]))

        for i in range(1, len(hidden_size)):
            self.layers.append(MiddleLayer(hidden_size[i - 1], hidden_size[i], bias=hidden_bias[i]))

        self.layers.append(OutputLayer(self.layers[-1].neurons_count, output_size, bias=output_bias))

    def forward_prop(self, X):
        for i in self.layers:
            X = i.calc_output(X)

        return X

    def back_prop(self, Y):
        self.layers[-1].calc_error(Y)

        for i in reversed(range(len(self.layers) - 1)):
            self.layers[i].calc_error(self.layers[i + 1])

    def update_weights(self, X, learning_rate):
        self.layers[0].update_weights(X, learning_rate)

        for i in range(1, len(self.layers)):
            self.layers[i].update_weights(self.layers[i - 1].output, learning_rate)

    def train(self, X, Y, learning_rate, max_epochs, min_error):
        self.clean()
        i = 0

        while (True):
            for j in range(len(X)):
                last = self.forward_prop(X[j])
                self.back_prop(Y[j])
                self.update_weights(X[j], learning_rate)

                if i % 1000 == 0:
                    self.save()

            if debug:
                if i % 100 == 0:
                    self.save()
                    print("Iteration: ", i, "Output:", last)

            if (i > max_epochs or self.check_min_error(min_error)):
                break

            i += 1

        self.save()

    def check_min_error(self, min_error):
        return np.mean(np.square(self.layers[-1].error)) < min_error

    def predict(self, X):
        result = self.forward_prop(X)
        original = result.copy()

        for j in range(len(result)):
            result[j] = 1 if result[j] >= 0.5 else 0

        return result, original

    def error(self, expected, predicted):
        return np.mean(np.square(expected - predicted))

    def clean(self):
        files = glob.glob('weights/*')
        for f in files:
            os.remove(f)

    def save(self):
        for i in range(len(self.layers)):
            name = "weights\\" + str(i) + ".txt"
            np.savetxt(name, self.layers[i].weights)

    def read(self):
        for i in range(len(self.layers)):
            name = "weights\\" + str(i) + ".txt"
            self.layers[i].weights = np.loadtxt(name)
