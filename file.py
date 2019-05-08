import numpy as np


def read_custom_test(name):
    data = []
    f = open(name, 'r')
    lines = f.readlines()

    init = 0
    final = 3

    data.append(get_number(init, final, lines))

    while (True):
        init, final = has_more(init, final, lines)

        if init == -1:
            return data
        else:
            data.append(get_number(init, final, lines))


def has_more(init, final, lines):
    for line in lines:
        if (len(line) - 1 > final):
            init = final + 1
            final += 4

            return init, final

    return -1, -1


def get_number(init, final, lines):
    count = 0
    result = [0] * 15

    for line in lines:
        for i in range(init, final):
            if (i >= len(line)):
                pass
            elif (line[i] == "X" or line[i] == "x"):
                result[count] = 1

            count += 1

    return result


def read_binary(name):
    return np.loadtxt(name)


def read_training(name):
    X = np.loadtxt(name, usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14))
    Y = np.loadtxt(name, usecols=(15, 16, 17, 18))

    return X, Y
