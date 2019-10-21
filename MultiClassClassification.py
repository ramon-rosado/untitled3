# Libraries
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from keras.datasets import mnist

np.random.seed(1212)


class AI():
    def __init__(self, alpha=0.01, max_iterations=5000, class_of_interest=0):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        self.alpha = alpha
        self.max_iterations = max_iterations
        self.class_of_interest = class_of_interest

    def sigmoid(self, x):
        return np.exp(x) / (np.exp(x) + 1)  # This is the normal sigmoid function

    def predict(self, x_bar, params):
        return self.sigmoid(np.dot(params, x_bar))

    def cost(self, x_train, y_train, params):
        cost = 0
        for x, y in zip(x_train, y_train):
            x_bar = np.array(np.insert(x, 0, 1))
            y_hat = self.predict(x_bar, params)

            if y == self.class_of_interest:
                y_binary = 1
            else:
                y_binary = 0
            cost += y_binary * np.log(y_hat) + (1 - y_binary) * np.log(1 - y_hat)
        return cost

    def train(self, x_train, label, print_iter = 5000):

        iteration = 1
        while iteration < self.max_iterations:

            for i, xy in enumerate(zip(x_train, label)):
                x_bar = np.array(np.insert(xy[0], 0, 1))
                y_hat = self.predict(x_bar, )