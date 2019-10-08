import numpy as np
import csv
import random


class NeuralNetwork():
    def __init__(self, neurons, hidden_acti,
                 output_acti):
        # arguments: an array "neurons" consist of number of neurons for each layer,
        # activation function to be used in hidden layers and activation function to be used in output layer
        self.inputSize = neurons[0]  # Number of neurons in input layer
        self.outputSize = neurons[-1]  # Number of neurons in output layer
        self.layers = len(neurons)
        self.w = []
        for i in range(len(neurons) - 1):
            self.w.append(
                np.random.normal(0, 0.1, size=neurons[i] * neurons[i + 1]).reshape([neurons[i], neurons[i + 1]]))
  
        self.activationHidden = None  # Activation funtion to be used in hidden layers
        self.activationOutput = None  # Activation funtion to be used in output layer
        self.activationHiddenPrime = None  # Derivative of the activation funtion to be used in hidden layers
        self.activationOutputPrime = None  # Derivative of the activation funtion to be used in output layer

        if (hidden_acti == "sigmoid"):
            self.activationHidden = self.sigmoid
            self.activationHiddenPrime = self.sigmoidPrime
        else:
            self.activationHidden = self.linear
            self.activationHiddenPrime = self.linearPrime

        if (output_acti == "sigmoid"):
            self.activationOutput = self.sigmoid
            self.activationOutputPrime = self.sigmoidPrime
        else:
            self.activationOutput = self.linear
            self.activationOutputPrime = self.linearPrime

    def sigmoid(self, s):  # sigmoid activation function
        return (1 / (1 + np.exp(-s)))

    def sigmoidPrime(self, x):  # derivative sigmoid activation function
        return (self.sigmoid(x) * (1 - self.sigmoid(x)))

    def linear(self, s):  # Linear activation function
        return (s)

    def linearPrime(self, x):  # derivative of linear activation function
        return (np.ones(len(x)))

    def forward(self, x):  # function of forward pass which will receive input and give the output of final layer
        Z1 = np.dot(self.w[0].T, x)
        A1 = self.sigmoid(Z1)
        Z2 = np.dot(self.w[1].T, A1)
        A2 = self.sigmoid(Z2)
        Z3 = np.dot(self.w[2].T, A2)
        A3 = self.sigmoid(Z3)
        Z4 = np.dot(self.w[3].T, A3)
        A4 = self.sigmoid(Z4)
        self.temp = {
            "Z1": Z1,
            "A1": A1,
            "Z2": Z2,
            "A2": A2,
            "Z3": Z3,
            "A3": A3,
            "Z4": Z4,
            "A4": A4
        }
        return A4

    def backward(self, x, y, o):  # find the loss and return derivative of loss w.r.t every parameter
        L = sum((o - y) ** 2)
        Z1 = self.temp["Z1"]
        A1 = self.temp["A1"]
        Z2 = self.temp["Z2"]
        A2 = self.temp["A2"]
        Z3 = self.temp["Z3"]
        A3 = self.temp["A3"]
        Z4 = self.temp["Z4"]
        A4 = self.temp["A4"]
        g4 = (2 * (o - y))
        g3 = np.matmul(np.diag(self.sigmoidPrime(Z3)), np.matmul(self.w[3], g4))
        g2 = np.matmul(np.diag(self.sigmoidPrime(Z2)), np.matmul(self.w[2], g3))
        g1 = np.matmul(np.diag(self.sigmoidPrime(Z1)), np.matmul(self.w[1], g2))
        dw3 = np.matmul(np.diag(self.sigmoidPrime(Z4)), np.multiply(g4, A4))
        dw2 = np.matmul(np.diag(self.sigmoidPrime(Z3)), np.multiply(g3, A3))
        dw1 = np.matmul(np.diag(self.sigmoidPrime(Z2)), np.multiply(g2, A2))
        dw0 = np.matmul(np.diag(self.sigmoidPrime(Z1)), np.multiply(g1, A1))

        grads = {
            "dw3": dw3,
            "dw2": dw2,
            "dw1": dw1,
            "dw0": dw0
        }
        return L, grads

    def update_parameters(self, grads, learning_rate):  # update the parameters using the gradients
        dw3 = grads["dw3"]
        dw1 = grads["dw1"]
        dw2 = grads["dw2"]
        dw0 = grads["dw0"]

        self.w[0] -= learning_rate * dw0
        self.w[1] -= learning_rate * dw1
        self.w[2] -= learning_rate * dw2
        self.w[3] -= learning_rate * dw3

    def train(self, X, Y):  # receive the full training data set
        lr = 1e-4  # learning rate
        epochs = 100  # number of epochs
        batchsize = 50
        batches = round(len(X)/batchsize)
        k = 0
        for e in range(epochs):
            sum1 = 0
            sumo = 0
            sumg = {
                "dw3": 0,
                "dw2": 0,
                "dw1": 0,
                "dw0": 0
            }
            for q in range(batches):
                sum1 = 0
                for a in range(batchsize):
                    out = self.forward(X[random.sample(range(0,24000),1)[0]])  # call of forward pass to get the predicted value
                    los, grads = self.backward(X[q], Y[q], out)  # find the gradients using backward pass
                    sumg["dw3"] += grads["dw3"]
                    sumg["dw2"] += grads["dw2"]
                    sumg["dw1"] += grads["dw1"]
                    sumg["dw0"] += grads["dw0"]
                    sum1 += los
                self.update_parameters(sumg, lr)
                sumo += sum1

            print('\nAverage Loss: ', sumo/len(X))

    def predict(self, x):
        print("Input : \n" + str(x))
        print("Output: \n" + str((self.forward(x))))


Y = []
X = []

with open('Assignment3_train_data.csv') as f:
    f_csv = csv.reader(f)
    headers = next(f_csv)
    for row in f_csv:

        if (row[0] == '1'):
            Y.append(np.array([1, 0, 0, 0]))
        if (row[0] == '2'):
            Y.append(np.array([0, 1, 0, 0]))
        if (row[0] == '3'):
            Y.append(np.array([0, 0, 1, 0]))
        if (row[0] == '4'):
            Y.append(np.array([0, 0, 0, 1]))

        X.append(row[1:])

for i in range(len(X)):
    for j in range(len(X[0])):
        X[i][j] = float(X[i][j]) / 255
    X[i] = np.array(X[i])

D_in, H1, H2, H3, D_out = len(X[0]), 300, 500, 300, 4

neurons = [D_in, H1, H2, H3, D_out]  # list of number of neurons in the layers sequentially.

Hidden_activation = "sigmoid"  # activation function of the hidden layers.
Output_activation = "sigmoid"  # activation function of the output layer.
test = NeuralNetwork(neurons, Hidden_activation, Output_activation)
test.train(X, Y)