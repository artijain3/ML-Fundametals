#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import argparse
import pandas as pd 
from matplotlib import pyplot as plt
import math

from operator import itemgetter


class Model(object):
    """
     Polynomial Regression.
    """

    def fit(self, X, y, k):
        """
        Fits the polynomial regression model to the training data.

        Arguments
        ----------
        X: nx1 matrix of n examples
        y: response variable vector for n examples
        k: polynomial degree
        """
        bias_function = np.zeros((len(X), k+1))

        for x in range(len(X)): 
            for i in range(k+1):
                x_value = X[x]**i
                bias_function[x][i] = x_value

        # print(bias_function.shape)

        biasTbias = np.dot(bias_function.T, bias_function)
        biasTy = np.dot(bias_function.T, y)
        inverse_bias = np.linalg.inv(biasTbias)
        res = np.dot(inverse_bias, biasTy)
        self.w = res



    def predict(self, X):
        """
        Predicts the dependent variable of new data using the model.

        Arguments
        ----------
        X: nx1 matrix of n examples

        Returns
        ----------
        response variable vector for n examples
        """
        # is my formula w^T (phi(x))
        k = len(self.w) 
        bias_function = np.zeros((len(X), k))

        for x in range(len(X)): 
            for i in range(k):
                x_value = X[x]**i
                bias_function[x][i] = x_value
        self.bias = bias_function

        y = np.dot(self.bias, self.w)
        # print(y.shape)
        return y
    

    def rmse(self, X, y):
        """
        Returns the RMSE(Root Mean Squared Error) when the model is validated.
        
        Arguments
        ----------
        X: nx1 matrix of n examples
        y: response variable vector for n examples
        
        Returns
        ----------
        RMSE when model is used to predict y
        """
        yhat = self.predict(X)
        substraction = np.subtract(yhat,y)
        squared = np.square(substraction)
        mean = squared.mean()
        return math.sqrt(mean)


#run command:
#python poly.py --data=data/poly_reg_data.csv

if __name__ == '__main__':

    #Read command line arguments
    parser = argparse.ArgumentParser(description='Fit a Polynomial Regression Model')
    parser.add_argument('--data', required=True, help='The file which contains the dataset.')
                        
    args = parser.parse_args()

    input_data = pd.read_csv(args.data)
    
    n = len(input_data['y'])
    n_train = 25
    n_val = n - n_train

    x = input_data['x']
    x_train = x[:n_train][:,None]
    x_val = x[n_train:][:,None]

    y= input_data['y']
    y_train = y[:n_train][:,None]
    y_val = y[n_train:][:,None]


    
    """
    4.3 # 1 
    Plot the training error (RMSE error on the training set) versus k. From the plot, which value
    of k gives you the minimum training error?
    """
    #plot training rmse versus k
    model = Model()

    all_rmse_values = []
    for k in range(1,11):
        model.fit(x_train, y_train, k)
        all_rmse_values.append(model.rmse(x_train, y_train))

    # find the min RMSE and corresponding k
    print(f"smallest rmse for training set is {min(all_rmse_values)}")
    smallest_k_for_training = all_rmse_values.index(min(all_rmse_values)) + 1
    print(f"the corresponding k is {smallest_k_for_training}")

    plt.title(f"RMSE error on training set vs. polynomial degree")
    plt.scatter(range(1,11), all_rmse_values) 
    plt.xlabel("Polynomial Degree")
    plt.ylabel("RMSE Error on Training Set")
    plt.show()
    
    """
    4.3 # 2
    Plot the validation error versus k. Which value of k gives you the minimum validation error?
    """
    all_rmse_values = []
    #plot validation rmse versus k
    for k in range(1,11):
        model.fit(x_train, y_train, k)
        all_rmse_values.append(model.rmse(x_val, y_val))

    # find the min RMSE and corresponding k
    print(f"smallest rmse for validation set is {min(all_rmse_values)}")
    smallest_k_for_validation = all_rmse_values.index(min(all_rmse_values)) + 1
    print(f"the corresponding k is {smallest_k_for_validation}")

    plt.title(f"RMSE error on validation set vs. polynomial degree")
    plt.scatter(range(1,11), all_rmse_values) 
    plt.xlabel("Polynomial Degree")
    plt.ylabel("RMSE Error on Validation Set")
    plt.show()

    
    """
    4.3 # 4
    For each k = 1, 3, 5, 10, create a scatter plot of the training data points. On the same plot, for each k, draw a line for the fitted polynomial.
    """
    #plot fitted polynomial curve versus k as well as the scattered training data points 
	# Your code here
    k_values = [1, 3, 5, 10]

    plt.scatter(x_train, y_train)
    #plt.show()

    colors = ['red', 'green', 'blue', 'purple']
    for k in range(len(k_values)):
        model.fit(x_train, y_train, k_values[k])

        # take the min and max of the x train and plot the stuff in between
        min_x = min(x_train)
        max_x = max(x_train)
        plt.title(f"Fitted K degress polynomial vs original training data")
        d = np.linspace(min_x, max_x, 100)
        y = model.predict(d)
        plt.plot(d, y, c=colors[k], label=f'k={k_values[k]}')
        plt.legend()
    plt.xlabel("Input Data, X")
    plt.ylabel("Output Prediction, Y")
    plt.show()



