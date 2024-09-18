"""Main script for the solution."""

import numpy as np
import pandas as pd
import argparse

import npnn
import numpyNN


def _get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lr", help="learning rate", type=float, default=0.1)
    p.add_argument("--opt", help="optimizer", default="SGD")
    p.add_argument(
        "--epochs", help="number of epochs to train", type=int, default=20)
    p.add_argument(
        "--save_stats", help="Save statistics to file", action="store_true")
    p.add_argument(
        "--save_pred", help="Save predictions to file", action="store_true")
    p.add_argument("--dataset", help="Dataset file", default="mnist.npz")
    p.add_argument(
        "--test_dataset", help="Dataset file (test set)",
        default="mnist_test.npz")
    p.set_defaults(save_stats=False, save_pred=False)
    return p.parse_args()


if __name__ == '__main__':
    args = _get_args()
    X, y = npnn.load_mnist(args.dataset)
    

    # TODO
    # Create dataset (see npnn/dataset.py)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    
    train_i = indices[10000:]
    val_i = indices[:10000]
    
    X_train = X[train_i]
    y_train = y[train_i]
    X_val = X[val_i]
    y_val = y[val_i]
    
    # print(f"X_train is : {X_train.shape}")
    # print(f"y_train is : {y_train.shape}")
    # print(f"X_val is : {X_val.shape}")
    # print(f"y_val is : {y_val.shape}")
    
    
    train_dataset = npnn.Dataset(X_train, y_train, 32)
    val_dataset = npnn.Dataset(X_val, y_val, 32)
    
    # Create model (see npnn/model.py)
    modules = [
        npnn.Flatten(),
        npnn.Dense(28*28, 256),
        npnn.ELU(0.9),
        npnn.Dense(256, 64),
        npnn.ELU(0.9),
        npnn.Dense(64, 10),   
    ]
    
    if args.opt == "SGD":
        opty = npnn.SGD(args.lr)
    else:
        opty = npnn.Adam(learning_rate=0.002, beta1=0.9, beta2=0.999, epsilon=1e-7, time_step=0)
        
    model = npnn.Sequential(modules=modules, loss=npnn.SoftmaxCrossEntropy(), optimizer=opty)
    
    # Train for args.epochs
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    for e in range(args.epochs):
        print(f"Starting epoch: {e}")
        loss_train, acc_train = model.train(train_dataset)
        train_losses.append(loss_train)
        train_accuracies.append(acc_train)
        
        val_loss, acc_val = model.test(val_dataset)
        val_losses.append(val_loss)
        val_accuracies.append(acc_val)
        
        # print(f"End of epoch {e}, train loss is: {loss_train}")
        # print(f"End of epoch {e}, train acc is: {acc_train}")
        # print(f"End of epoch {e}, val loss is: {val_loss}")
        # print(f"End of epoch {e}, val acc is: {acc_val}")
        
    stats = pd.DataFrame()
    stats["training_loss"] = train_losses
    stats["train_accuracies"] = train_accuracies
    stats["val_losses"] = val_losses
    stats["val_accuracies"] = val_accuracies
    
    # Printing the final losses and accuracies
    print("Final Training Loss:")
    print(f"train_loss = {train_losses}")
    
    print("\nFinal Val Loss:")
    print(f"val_loss = {val_losses}")
    
    print("\nFinal Train Accuracy:")
    print(f"train_acc = {train_accuracies}")
    
    print("\nFinal Val Accuracy:")
    print(f"val_acc = {val_accuracies}")
    
    # Save statistics to file.
    # We recommend that you save your results to a file, then plot them
    # separately, though you can also place your plotting code here.
    if args.save_stats:
        stats.to_csv("data/{}_{}.csv".format(args.opt, args.lr))

    # Save predictions.
    if args.save_pred:
        X_test, _ = npnn.load_mnist("mnist_test.npz")
        y_pred = np.argmax(model.forward(X_test), axis=1).astype(np.uint8)
        np.save("mnist_test_pred.npy", y_pred)
