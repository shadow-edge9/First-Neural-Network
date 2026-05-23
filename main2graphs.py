#!/usr/bin/env python3
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

x = np.array(([0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]))
y = (([1], [0], [0], [0], [0], [0], [0], [1]))

xPredicted = np.array(([0, 0, 1]))  # change this input to get the output for different inputs
x = x / np.amax(x, axis=0)

lossFile = open("SumSquaredLossList.csv", "w")


class neuralNetwork(object):
    def __init__(self):
        # paramters:
        self.inputLayers = 3
        self.outputLayers = 1
        self.hiddenLayers = 4

        self.w1 = np.random.randn(self.inputLayers, self.hiddenLayers)
        self.w2 = np.random.randn(self.hiddenLayers, self.outputLayers)

    def activationSigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def activationSigmoidPrime(self, z):
        return z * (1 - z)

    def feedForward(self, x):
        self.z = np.dot(x, self.w1)
        self.z2 = self.activationSigmoid(self.z)

        self.z3 = np.dot(self.z2, self.w2)
        o = self.activationSigmoid(self.z3)

        return o

    def backwardPropagation(self, x, y, o):
        self.o_error = y - o
        self.o_delta = self.o_error * self.activationSigmoidPrime(o)
        self.z2_error = self.o_delta.dot(self.w2.T)
        self.z2_delta = self.z2_error * self.activationSigmoidPrime(self.z2)

        self.w1 += x.T.dot(self.z2_delta)
        self.w2 += self.z2.T.dot(self.o_delta)

    def trainNetwork(self, x, y):
        o = self.feedForward(x)
        self.backwardPropagation(x, y, o)

    def saveSumSquaredLossList(self, i, error):
        lossFile.write(str(i) + "" + str((error).tolist()) + "\n")

    def saveWeights(self):
        np.savetxt("weightsLayers2.txt", self.w2, fmt="%s")

    def predictOutput(self):
        print("Predicted Output on trained weights: ")
        print("Expected(x1-x3): \n" + str(xPredicted))
        print("Output Y1: \n" + str(self.feedForward(xPredicted)))


myneuralNetwork = neuralNetwork()
trainingEpochs = 1000
# trainingEpochs = 10000

for i in range(trainingEpochs):
    print("Epoch: " + str(i) + "\n")
    print("Network Input: " + str(x))
    print("Expected: " + str(y))
    print("Actual: " + str(myneuralNetwork.feedForward(x)))

    Loss = np.mean(np.square(y - myneuralNetwork.feedForward(x)))
    myneuralNetwork.saveSumSquaredLossList(i, Loss)
    print("Sum squared Loss: " + str(Loss))
    print("\n")
    myneuralNetwork.trainNetwork(x, y)

myneuralNetwork.saveWeights()
myneuralNetwork.predictOutput()

#################################################################
print("\033[93m= = = = = = T R A I N I N G    D Y N A M I C S = = = = = =\033[0m\n")
loss_df = pd.read_csv("SumSquaredLossList.csv", header=None, names=["Epoch Loss"])
loss_df["Epoch"] = loss_df.index

print("= = = = = L O S S   C U R V E = = = = =")
sns.lineplot(x="Epoch", y="Epoch Loss", data=loss_df)
plt.title("Training Loss over Epochs")
plt.show()

predictions = myneuralNetwork.feedForward(x)
print(type(predictions))
results_df = pd.DataFrame({"Expected": np.array(y).flatten(), "Predicted": predictions.flatten()})

print("= = = = = P R E D I C T I O N S   VS  T R U E   L A B E L S = = = = =")
sns.scatterplot(x="Expected", y="Predicted", data=results_df)
plt.title("Expected vs Predicted Outputs")
plt.show()

sns.histplot(results_df["Predicted"], kde=True)
plt.title("Distribution of Predicted Outputs")
plt.show()

print("= = = = = W E I G H T   A N A L Y S I S = = = = =")
sns.heatmap(myneuralNetwork.w1, annot=True, cmap="coolwarm")
plt.title("Weights of Layer 1")
plt.show()

sns.heatmap(myneuralNetwork.w2, annot=True, cmap="coolwarm")
plt.show()


