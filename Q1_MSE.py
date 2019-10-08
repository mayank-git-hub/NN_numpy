import torch
import csv
from tqdm import tqdm
import numpy as np


class NeuralNetwork:
    def __init__(self, neurons, hidden_acti,
                 output_acti):
        # arguments: an array "neurons" consist of number of neurons for each layer,
        # activation function to be used in hidden layers and activation function to be used in output layer
        self.inputSize = neurons[0]  # Number of neurons in input layer
        self.outputSize = neurons[-1]  # Number of neurons in output layer
        self.layers = len(neurons)
        self.w = []

        for i in range(len(neurons) - 1):

            cur_weight = torch.from_numpy(
                    np.random.normal(0, 0.1, size=neurons[i] * neurons[i + 1]).reshape([neurons[i], neurons[i + 1]]))

            if torch.cuda.is_available():
                cur_weight = cur_weight.cuda()

            self.w.append(cur_weight)
  
        self.activationHidden = None  # Activation funtion to be used in hidden layers
        self.activationOutput = None  # Activation funtion to be used in output layer
        self.activationHiddenPrime = None  # Derivative of the activation funtion to be used in hidden layers
        self.activationOutputPrime = None  # Derivative of the activation funtion to be used in output layer

        if hidden_acti == "sigmoid":
            self.activationHidden = self.sigmoid
            self.activationHiddenPrime = self.sigmoidPrime
        else:
            self.activationHidden = self.linear
            self.activationHiddenPrime = self.linearPrime

        if output_acti == "sigmoid":
            self.activationOutput = self.sigmoid
            self.activationOutputPrime = self.sigmoidPrime
        else:
            self.activationOutput = self.linear
            self.activationOutputPrime = self.linearPrime

    def sigmoid(self, s):  # sigmoid activation function
        return 1 / (1 + torch.exp(-s))

    def sigmoidPrime(self, x):  # derivative sigmoid activation function
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def linear(self, s):  # Linear activation function
        return s

    def linearPrime(self, x):  # derivative of linear activation function
        return torch.ones(len(x))

    def forward(self, x):  # function of forward pass which will receive input and give the output of final layer

        Z1 = torch.matmul(self.w[0].transpose(1, 0), x)
        # print(Z1.shape, 'Z1')
        A1 = self.sigmoid(Z1)
        # print(A1.shape, 'A1')
        Z2 = torch.matmul(self.w[1].transpose(1, 0), A1)
        # print(Z2.shape, 'Z2')
        A2 = self.sigmoid(Z2)
        # print(A2.shape, 'A2')
        Z3 = torch.matmul(self.w[2].transpose(1, 0), A2)
        # print(Z3.shape, 'Z3')
        A3 = self.sigmoid(Z3)
        # print(A3.shape, 'A3')
        Z4 = torch.matmul(self.w[3].transpose(1, 0), A3)
        # print(Z4.shape, 'Z4')
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

    def backward(self, y, o):  # find the loss and return derivative of loss w.r.transpose(1, 0) every parameter

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
        g3 = torch.mul(self.sigmoidPrime(Z3), torch.matmul(self.w[3], g4))
        g2 = torch.mul(self.sigmoidPrime(Z2), torch.matmul(self.w[2], g3))
        g1 = torch.mul(self.sigmoidPrime(Z1), torch.matmul(self.w[1], g2))
        dw3 = torch.mul(self.sigmoidPrime(Z4), torch.mul(g4, A4))
        dw2 = torch.mul(self.sigmoidPrime(Z3), torch.mul(g3, A3))
        dw1 = torch.mul(self.sigmoidPrime(Z2), torch.mul(g2, A2))
        dw0 = torch.mul(self.sigmoidPrime(Z1), torch.mul(g1, A1))

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

        lr = 1e-3  # learning rate

        epochs = 100
        # number of epochs
        batchsize = 50
        batches = round(len(X)/batchsize)
        for e in range(epochs):
            sumo = 0
            batch_iterator = tqdm(range(batches))

            for q in batch_iterator:

                item = np.random.randint(24000, size=batchsize)
                out = self.forward(X[item].transpose(1, 0))
                los, grads = self.backward(Y[item].transpose(1, 0), out)

                los = torch.mean(los)
                for grad_i in grads:
                    grads[grad_i] = torch.mean(grads[grad_i], dim=1)

                self.update_parameters(grads, lr)

                sumo += los.item()

                batch_iterator.set_description('Average Loss: ' + str(sumo/(q + 1)))

            print('\nAverage Loss: ', sumo/len(X))

    def predict(self, x):
        print("Input : \n" + str(x))
        print("Output: \n" + str((self.forward(x))))


if __name__ == "__main__":

    np.random.seed(0)
    Y = np.zeros([24000, 4])
    X = np.zeros([24000, 784])

    with open('Assignment3_train_data.csv') as f:
        f_csv = csv.reader(f)
        headers = next(f_csv)
        for no, row in enumerate(f_csv):

            Y[no, int(row[0]) - 1] = 1
            X[no, :] = np.array(row[1:]).astype(np.float32)/255

    D_in, H1, H2, H3, D_out = len(X[0]), 300, 500, 300, 4

    neurons = [D_in, H1, H2, H3, D_out]  # list of number of neurons in the layers sequentially.

    Hidden_activation = "sigmoid"  # activation function of the hidden layers.
    Output_activation = "sigmoid"  # activation function of the output layer.
    test = NeuralNetwork(neurons, Hidden_activation, Output_activation)

    X = torch.from_numpy(X)
    Y = torch.from_numpy(Y)

    if torch.cuda.is_available():

        X = X.cuda()
        Y = Y.cuda()

    test.train(X, Y)
