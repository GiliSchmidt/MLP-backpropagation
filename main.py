import network as n
import datetime
import file as f


def main():
    # ----------------------------- INITIALIZE NETWORK -----------------------------------------
    """
    input_size - size of layer Input
    hidden_size[] - array containing the size of each hidden layer
    output_size - size of layer Output
    hidden_bias[] -array of bias for each hidden layer - NULLABLE
    output_bias - output bias - NULLABLE
    """
    net = n.Network(15, [15], 4)

    # ----------------------------- TRAINING -----------------------------------------
    # TRAINING: will train test and test it
    # training settings can be modified insite the method

    #train(net)

    # ----------------------------- PREDICT -----------------------------------------
    # if NOT testing, get weights from previous training
    # then choose one file, custom or in binary

    net.read()
    predict_array(f.read_custom_test('data\custom_all.txt'), net)
    # predict_array(f.read_binary('data/binary_all.txt'), net)


def predict_array(X, net):
    for i in X:
        predict(i, net)


def predict(X, net):
    res = net.predict(X)
    print("Input:", X, "Result:", res[0], "Accuracy:", res[1])


def train(net):
    file = "train/numers_plus_all.txt"
    training_rate = 0.25
    epochs = 10000
    min_error = 0.00001

    X, Y = f.read_training(file)

    print("Started training:")
    time()

    net.train(X, Y, training_rate, epochs, min_error)

    print("Done training.")
    time()
    print("")

    test(net, X, Y)


def test(net, X, Y):
    aux = []

    print("Testing...")

    for i in range(len(Y)):
        res = net.predict(X[i])
        aux.append(res[0])
        print("Input:", X[i], "Expected:", Y[i], "Result:", res[0], "Error: ",
              net.error(Y[i], res[0]), "Accuracy: ", res[1])

    print("Total error:", net.error(Y, aux))


def time():
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))


main()
