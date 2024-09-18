"""Neural Network model."""

from .modules import Module
from .optimizer import Optimizer
from tqdm import tqdm
import numpy as np


def categorical_cross_entropy(pred, labels, epsilon=1e-10):
    """Cross entropy loss function.

    Parameters
    ----------
    pred : np.array
        Softmax label predictions. Should have shape (dim, num_classes).
    labels : np.array
        One-hot true labels. Should have shape (dim, num_classes).
    epsilon : float
        Small constant to add to the log term of cross entropy to help
        with numerical stability.

    Returns
    -------
    float
        Cross entropy loss.
    """
    assert(np.shape(pred) == np.shape(labels))
    return np.mean(-np.sum(labels * np.log(pred + epsilon), axis=1))


def categorical_accuracy(pred, labels):
    """Accuracy statistic.

    Parameters
    ----------
    pred : np.array
        Softmax label predictions. Should have shape (dim, num_classes).
    labels : np.array
        One-hot true labels. Should have shape (dim, num_classes).

    Returns
    -------
    float
        Mean accuracy in this batch.
    """
    assert(np.shape(pred) == np.shape(labels))
    return np.mean(np.argmax(pred, axis=1) == np.argmax(labels, axis=1))


class Sequential:
    """Sequential neural network model.

    Parameters
    ----------
    modules : Module[]
        List of modules; used to grab trainable weights.
    loss : Module
        Final output activation and loss function.
    optimizer : Optimizer
        Optimization policy to use during training.
    """

    def __init__(self, modules, loss=None, optimizer=None):

        for module in modules:
            assert(isinstance(module, Module))
        assert(isinstance(loss, Module))
        assert(isinstance(optimizer, Optimizer))

        self.modules = modules
        self.loss = loss

        self.params = []
        for module in modules:
            self.params += module.trainable_weights

        self.optimizer = optimizer
        self.optimizer.initialize(self.params)

    def forward(self, X, train=True):
        """Model forward pass.

        Parameters
        ----------
        X : np.array
            Input data

        Keyword Args
        ------------
        train : bool
            Indicates whether we are training or testing.

        Returns
        -------
        np.array
            Batch predictions; should have shape (batch, num_classes).
        """
        x_copy = X.copy()
        for module in self.modules:
            x_copy = module.forward(x_copy, train=train)
        y_predict = self.loss.forward(x_copy)
        return y_predict

    def backward(self, y):
        """Model backwards pass.

        Parameters
        ----------
        y : np.array
            True labels.
            
        Calcuate the loss on y and then for all the modules, send it backwards
        """
        y = self.loss.backward(y) # calculating the loss -> this will be a module we pass in
        
        num_modules = len(self.modules)
        for i in range(num_modules):
            y = self.modules[num_modules-1].backward(y)
            num_modules -= 1
        return y

    def train(self, dataset):
        """Fit model on dataset for a single epoch.

        Parameters
        ----------
        X : np.array
            Input images
        dataset : Dataset
            Training dataset with batches already split.

        Notes
        -----
        You may find tqdm, which creates progress bars, to be helpful:

        Returns
        -------
        (float, float)
            [0] Mean train loss during this epoch.
            [1] Mean train accuracy during this epoch.
        """
        mean_train_loss_list = []
        mean_train_accuracy_list = []
        for X,y in tqdm(dataset, desc="Training Loop"): #
            out = self.forward(X)
            
            loss = categorical_cross_entropy(out, y)
            
            grad = self.backward(y)
            mean_train_loss_list.append(loss)
            mean_train_accuracy_list.append(categorical_accuracy(out, y))

            final = self.optimizer.apply_gradients(self.params)
            
            
        mean_train_loss = np.sum(mean_train_loss_list)/len(mean_train_loss_list)
        mean_train_accuracy = np.sum(mean_train_accuracy_list)/len(mean_train_accuracy_list)
        return(mean_train_loss,mean_train_accuracy)

    def test(self, dataset):
        """Compute test/validation loss for dataset.

        Parameters
        ----------
        dataset : Dataset
            Validation dataset with batches already split.

        Returns
        -------
        (float, float)
            [0] Mean test loss.
            [1] Test accuracy.
        """
        mean_train_loss_list = []
        mean_train_accuracy_list = []
        for X,y in tqdm(dataset, desc="Testing Loop"): #
            pred = self.forward(X)
            loss = categorical_cross_entropy(pred, y)
            mean_train_loss_list.append(loss)
            mean_train_accuracy_list.append(categorical_accuracy(pred, y))
        
        mean_train_loss = np.sum(mean_train_loss_list)/len(mean_train_loss_list)
        mean_train_accuracy = np.sum(mean_train_accuracy_list)/len(mean_train_accuracy_list)
        return(mean_train_loss,mean_train_accuracy)
