"""
Variational Quantum Classifier
"""
from typing import *

import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer

# create a quantum device with 4 wires or qubits
dev = qml.device("default.qubit", wires=4)


def layer(W):
    """
    Block of our variational circuit
    :param W:
    :return:
    """

    # perform arbitrary rotation on every qubit
    qml.Rot(W[0, 0], W[0, 1], W[0, 2], wires=0)
    qml.Rot(W[1, 0], W[1, 1], W[1, 2], wires=1)
    qml.Rot(W[2, 0], W[2, 1], W[2, 2], wires=2)
    qml.Rot(W[3, 0], W[3, 1], W[3, 2], wires=3)

    # entangle each wire to its neighbor
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[3, 0])


def state_preparation(x):
    """
    encode our data inputs into the circuit
    :param x:
    :return:
    """
    qml.BasisState(x, wires=[0, 1, 2, 3])


@qml.qnode(dev)
def circuit(weights, x):
    """
    The quantum circuit or function we are trying to optimize
    :param weights:
    :param x:
    :return:
    """
    # encode our data into the circuit, expects inputs to be a list of 1s and 0s
    state_preparation(x)

    for W in weights:
        layer(W)

    return qml.expval(qml.PauliZ(0))


def variational_classifier(var, x):
    """
    Variational classifier using classical bias
    :param var:
    :param x:
    :return:
    """
    weights = var[0]
    bias = var[1]
    return circuit(weights, x) + bias


def mean_square_loss(labels, predictions):
    """
    Loss function for Variational Quantum classifier
    :param labels:
    :param predictions:
    :return:
    """
    loss = 0
    for label, prediction in zip(labels, predictions):
        loss = loss + (label - prediction) ** 2

    return loss / len(labels)


def accuracy(labels, predictions):
    """
    Accuracy for VQC
    :param labels:
    :param predictions:
    :return:
    """
    accuracy_ = 0
    for label, prediction in zip(labels, predictions):
        if abs(label - prediction) < 1e-5:
            accuracy_ += 1

    return accuracy_ / len(labels)


def cost(var, inputs, targets):
    """
    VQC cost function
    :param var:
    :param inputs:
    :param targets:
    :return:
    """
    predictions = [variational_classifier(var, x) for x in inputs]
    return mean_square_loss(targets, predictions)


def load_data(
        fp: str = "data/parity.txt",
        verbose: bool = False,
        num_examples_to_print: int = 5
) -> Tuple[np.array, np.array]:
    """
    Loads data for classification tasks
    :param fp: filepath to data
    :param verbose: print the data
    :param num_examples_to_print
    :return:
    """
    data = np.loadtxt(fp)
    X = np.array(data[:, :-1], requires_grad=False)
    Y = np.array(data[:, -1], requires_grad=False)
    Y = Y * 2 - np.ones(len(Y))  # shift label from {0, 1} to {-1, 1}

    if verbose:
        for i in range(num_examples_to_print):
            print("X = {}, Y = {: d}".format(X[i], int(Y[i])))

    return X, Y


np.random.seed(0)
num_qubits = 4
num_layers = 2
var_init = (0.01 * np.random.randn(num_layers, num_qubits, 3), 0.0)

print(var_init)

opt = NesterovMomentumOptimizer(0.5)
batch_size = 5

