#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import argparse
import math

from operator import itemgetter

"""
Matrix Y -> concentration of glucose and ethanol for n = 166 alcoholic fermentation

2 dependent variables
- 255 covariates in X 
    - Contain the first derivatives NIR (1115-2285)

Predict the glucose concentration for the given covariances

The training dataset has 126 observations
Valicdation set and testing set - 20 observations each
"""


"""
to get w0, do the fit without lambda I and just take the w0 from that

"""

class Model(object):
    """
     Ridge Regression.
    """

    def fit(self, X, y, alpha=0):
        """
        Fits the ridge regression model to the training data.

        Arguments
        ----------
        X: nxp matrix of n examples with p independent variables
        y: response variable vector for n examples
        alpha: regularization parameter.
        """

        # add the ones to the X
        n,p = X.shape
        ones = np.ones((n, 1))
        X_with_ones = np.hstack((ones, X))
                      
        # calculate
        XTX = np.dot(X_with_ones.T, X_with_ones)
        XTy = np.dot(X_with_ones.T, y)
        n, p = XTX.shape # n and p should be the same because we added the column of ones
        alpha_identity = np.identity(p) # identity matrix
        alpha_identity[0][0] = 0 # so u dont regularize the bias term
        beta = np.dot(np.linalg.inv((XTX + alpha*alpha_identity)), XTy)    
        self.bias = beta[0] #bias term 
        self.beta = beta[1:] #saving the beta without the bias term

       	
    def predict(self, X):
        """
        Predicts the dependent variable of new data using the model.

        Arguments
        ----------
        X: nxp matrix of n examples with p covariates

        Returns
        ----------
        response variable vector for n examples
        """
       	# Your code here
        # y = Beta^transpose X         
        y = self.bias +  np.dot(X, self.beta)
        return y


    def rmse(self, X, y):
        """
        Returns the RMSE(Root Mean Squared Error) when the model is validated.
            
        Arguments
        ----------
        X: nxp matrix of n examples with p covariates
        y: response variable vector for n examples
            
        Returns
        ----------
        RMSE when model is used to predict y
        """
       	# Your code here
        yhat = self.predict(X) 
        substraction = np.subtract(yhat,y)
        squared = np.square(substraction)
        mean = squared.mean()
        return math.sqrt(mean)


#run command:
#python ridge.py --X_train_set=data/Xtraining.csv --y_train_set=data/Ytraining.csv --X_val_set=data/Xvalidation.csv --y_val_set=data/Yvalidation.csv --y_test_set=data/Ytesting.csv --X_test_set=data/Xtesting.csv
if __name__ == '__main__':

    #Read command line arguments
    parser = argparse.ArgumentParser(description='Fit a Ridge Regression Model')
    parser.add_argument('--X_train_set', required=True, help='The file which contains the covariates of the training dataset.')
    parser.add_argument('--y_train_set', required=True, help='The file which contains the response of the training dataset.')
    parser.add_argument('--X_val_set', required=True, help='The file which contains the covariates of the validation dataset.')
    parser.add_argument('--y_val_set', required=True, help='The file which contains the response of the validation dataset.')
    parser.add_argument('--X_test_set', required=True, help='The file which containts the covariates of the testing dataset.')
    parser.add_argument('--y_test_set', required=True, help='The file which containts the response of the testing dataset.')
                        
    args = parser.parse_args()

    #Parse training dataset
    X_train = np.genfromtxt(args.X_train_set, delimiter=',')
    y_train = np.genfromtxt(args.y_train_set,delimiter=',')
    
    #Parse validation set
    X_val = np.genfromtxt(args.X_val_set, delimiter=',')
    y_val = np.genfromtxt(args.y_val_set, delimiter=',')
    
    #Parse testing set
    X_test = np.genfromtxt(args.X_test_set, delimiter=',')
    y_test = np.genfromtxt(args.y_test_set, delimiter=',')
    

    #FIND THE BEST REGULARIZER TERM
	# for each lambda: {a · 10^b : a ∈ {1,2,··· ,9},b ∈ {−5,−4,··· ,0}}.
    # calculating lambda values:
    saving_beta_values = []

    lambda_values = []
    for a in [1,2, 3, 4, 5, 6, 7, 8, 9]:
        for b in [-5,-4, -3, -2, -1, 0]:
            lambda_values.append(a * (10**b))

    # getting each beta
    model = Model()
    colors =  ['red', 'green', 'blue', 'yellow', 'purple', 'magenta', 'sienna', 'hotpink', 'firebrick', 'deepskyblue']

    for l in lambda_values:
        model.fit(X_train, y_train, alpha=l) #passing in the different values of l
        ten_beta_values = model.beta[0:10]

        for val in range(len(ten_beta_values)): 
            plt.scatter(l, ten_beta_values[val], c=colors[val]) 
            plt.xscale("log")
    plt.title(f"10 Learned Coefficients Beta with respect to Lambda")
    plt.xlabel("Lambda")
    plt.ylabel("Beta values")
    plt.show()
        
    
    #PLOT RMSE VERSUS LAMBDA
    """
    Plot on the y-axis the RMSE (Root Mean Squared Error) of the learned model on the validation set with respect to the regularization parameter. 
    Find the regularization parameter λ∗ that achieves the minimum RMSE.
    """
	# Your code here
        # first do the test set and get your ytrain
        # then subtract yval - ytrain
    # First get your beta

    all_rmse_values = []

    model = Model()
    for l in lambda_values:
        model.fit(X_train, y_train, alpha=l) #passing in the different values of l
        rmse = model.rmse(X_val, y_val)
        all_rmse_values.append(rmse)
        plt.scatter(l, rmse, c="blue")
        plt.xscale("log")
    plt.title(f"RMSE for validation set with respect to Lambda Values")
    plt.xlabel("Lambda")
    plt.ylabel("RMSE values")
    plt.show()

    # Finding the minimum RMSE value and the corresponding lambda
    print(f"smallest rmse is {min(all_rmse_values)}")
    print(f"the correspondig lambda is: {lambda_values[all_rmse_values.index(min(all_rmse_values))]}")


    #plot predicted versus real value
    """
    For the λ∗ found in Part 2 above, plot the predicted versus the real value of the glucose concentration, when the model is evaluated on the testing dataset. 
    That is, for the 20 testing points, make a scatter plot of the true values of glucose concentration (on x-axis) vs. the predicted values of glucose concentration (on y-axis).
    """
	# Your code here
    lambda_val = lambda_values[all_rmse_values.index(min(all_rmse_values))] # getting the ideal lambda
    model = Model()
    model.fit(X_train, y_train, alpha=lambda_val)
    y = model.predict(X_test)
    print(y)
    plt.scatter(y_test, y)
    plt.title(f"True Values of Glucose Concentration vs. Predicted Values of Glucose Concentration")
    plt.xlabel("True Values of Glucose Concentration")
    plt.ylabel("Predicted Values of Glucose Concentration")
    plt.show()
    


    #plot regression coefficients - use training data set
    """
    L = 20
    Nsum = 50
    N = len(validation set)
    1. make data sets
    2. for each data set, use the sample to get ynl
    3. for each data set, calculate E[yn] which is ynbar  -> the yn in this equation is given
    """

    np.random.seed(3)
	# Your code here
    model = Model()

    # setting variables
    Ldatasets = 20
    Nsub = 50

    final_var = []


    new_X_list = []
    new_Y_list = []

    for l in range(Ldatasets):
        indices = np.random.choice(X_train.shape[0], Nsub)
        new_X =np.array(X_train[indices, :]) # getting your new X (sampled)
        new_Y = np.array(y_train[indices]) # getting my new y (sampled)

        N =  X_val.shape[0]
        new_X_list.append(new_X)
        new_Y_list.append(new_Y)
    
    for lambda_val in lambda_values:
        yhats_per_L = [] # all the rando y predictions, 20 50x1 vectors


        for l in range(Ldatasets):
            new_X = new_X_list[l]
            new_Y = new_Y_list[l]
            model.fit(new_X, new_Y, alpha=lambda_val) #passing in the different values of l
            yhat = model.predict(X_val)
            yhats_per_L.append(yhat) # saving all my yhats
  
        # calculating the expected value
        y_bar = [0] * N
        for l in range(Ldatasets):
            for n in range(N):
                y_bar[n] += yhats_per_L[l][n]

        for n in range(len(y_bar)):
            y_bar[n] =  y_bar[n]/Ldatasets


        # calculating the variance 
        var_for_lambda = 0
        total_sum = 0
        for l in range(Ldatasets):
            subtracted_y = yhats_per_L[l] - y_bar
            total_sum += np.dot(subtracted_y.T,subtracted_y) # takes care of the extra summation and square

        total_sum = total_sum/(N*Ldatasets)
        final_var.append(total_sum)

    plt.figure()
    plt.scatter(lambda_values, final_var)
    plt.title(f"Bias Variance Tradeoff")
    plt.xlabel("Lambda")
    plt.ylabel("Variance")
    plt.xscale('log')
    plt.show()