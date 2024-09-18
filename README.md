# ML-Fundametals
This repository contains all numpy implementations of fundamental ML algorithms: linear regression, logistic regression,  

## Linear Regression
This contains an implementation of a ridge regression estimator and polynomial regression. For both, there are methods to fit the model given a value for the regularization parameter, a function to compute the RMSE error, and a function to predict the values of the dependent variable. 

## Logistic Regression
This directory contains an implementation for unregularized and regularized logistic regression for binary classification problems using gradient descent. This includes a method for feature representation as well as code to train and perform optimization.

## SVM
This contains an implementation of a linear SVM. There are functions to pre-process the data, train and test the SVM, and perform cross-validation.

## Adaboost
This contains an Adaboost algorithm implementation with decision strumps.

## Gaussian Mixture Model (GMM)
This contains an implementation of GMMs to classify a dataset of 3000 unique examples with K components, generated from a normal distribution (satisfying the assumptions of a Gaussian mixture model). There is also an implementation of the EM algorithm.

## Neural Network
This directory contains a numpy implementation of a neural network to classify the MNIST dataset. This includes the architecture for the multi-layered perceptron, forward and backward passes for sfotmax cross entropy, exponential linear unit (ELU) activation, and a dense (fully connected) layer. This bare bones implementation was then used to built a neural network training architecture.

## PCA
This directory contains an implementation of PCA to perform dimensionality reduction on faces from the Yale Face Database B.

## Decision Tree
This contains a decision tree classifier using scikit-learn and using it to classify the Spotify data set which contains information like a list of songs, features about the song, and whether the individual likes or disliked the song. The implemented decision tree can be used to predict whether this individual would like or dislike a song based on a list of features. 

## RL
This is an implementation of value iteration in Frozen-lake environment (from OpenAI Gym). Frozen lake is a 4x4 or 8x8 grid-world environment where the agent can move up, down, right, and left to reach the goal and avoid falling into holes. 

