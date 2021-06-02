from typing import *
from argparse import ArgumentParser

import pennylane as qml
from pennylane import numpy as np  # we want to import the pennylane numpy not regular numpy

# create a PennyLane device
# wires - is the name of subsystems the device will be initialized with
dev1 = qml.device("default.qubit", wires=1)


# define our quantum function
@qml.qnode(dev1)
def circuit(params: Union[List, Tuple]):
    """
    Creates a quantum function or circuit, where first we rotate the qubit around the x-axis
    and then about the y-axis. We then will measure the expectation of the Pauli-Z operator.
    < ψ | σ | ψ >

    :param params:
    :return:
    """
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=0)
    return qml.expval(qml.PauliZ(0))


def quantum_gradient_function(
        q_function,
        argnum: Union[int, List[int], Tuple[int]]
):
    """
    Computes quantum gradients for a quantum function q_function
    :param q_function:
    :param argnum:
    :return:
    """
    return qml.grad(q_function, argnum=argnum)


def cost(x):
    """
    Cost function used to optimize the qubit rotation circuit
    :return:
    """
    return circuit(x)


def main(args):
    """
    Main script function that performs optimization to find values of phi1 and phi2 such that a
    qubit is rotated from initial state |0> to final state |1>. This is equivalent to measuring the expectation of
    the Pauli-Z operator to be -1.

    :param args:
    :return:
    """
    params = [args.phi1, args.phi2]
    optimizer = qml.GradientDescentOptimizer(args.step_size)
    loss_history = np.zeros(args.n_steps + 1)
    loss_history[0] = cost(params)
    for i in range(args.n_steps):
        loss_history[i + 1] = cost(params)
        params = optimizer.step(cost, params)
        if i % 5 == 0:
            print(f"Cost after step {i} is {cost(params): .5f}.")

    print(f"Optimized rotation angles: phi1 = {params[0]}, phi2 = {params[1]}. Final loss: {cost(params)}.")

    # circuit_value = circuit([args.phi1, args.phi2])
    # d_circuit = quantum_gradient_function(circuit, argnum=0)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--phi1", default=0.012, help="Initial value for angle of rotation about x-axis")
    parser.add_argument("--phi2", default=0.011, help="Initial value for angle of rotation about y-axis")
    parser.add_argument("--step_size", default=0.4, help="Optimizer step size used in the optimization process.")
    parser.add_argument("--n_steps", default=100, help="Number of optimization steps to take.")

    args = parser.parse_args()
    main(args)






