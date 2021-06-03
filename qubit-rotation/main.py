from typing import *
from argparse import ArgumentParser
import math

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import seaborn as sns
import pennylane as qml
from pennylane import numpy as np  # we want to import the pennylane numpy not regular numpy

sns.set()

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
    :return: a value in the range [-1, 1] - the Pauli-Z expected value
    """
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=0)
    return qml.expval(qml.PauliZ(0))


# define our quantum function
@qml.qnode(dev1)
def circuit_positional(phi1: float, phi2: float):
    """
    Creates a quantum function or circuit, where first we rotate the qubit around the x-axis
    and then about the y-axis. We then will measure the expectation of the Pauli-Z operator.
    < ψ | σ | ψ >

    :param phi1: angle of rotation about x-axis
    :param phi2: angle of rotation about y-axis
    :return: a value in the range [-1, 1] - the Pauli-Z expected value
    """
    qml.RX(phi1, wires=0)
    qml.RY(phi2, wires=0)
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


def cost_positional(phi1, phi2):
    """
    Cost function used to optimize the qubit rotation circuit, but with positional arguments rather than an array
    of parameters
    :return:
    """
    return circuit_positional(phi1, phi2)


def plot_cost_vs_rotation_angles(
        phi1_range: Tuple = (-2 * math.pi, 2 * math.pi),
        phi2_range: Tuple = (-2 * math.pi, 2 * math.pi),
        step: int = 0.2
):

    # Can't get meshgrid to work with this due to shape errors
    # TODO look into removing the loop with numpy.meshgrid or similar

    phi1_values = np.arange(*phi1_range, step=step)
    phi2_values = np.arange(*phi2_range, step=step)
    cost_values = np.zeros(phi1_values.shape[0])

    for i, (phi1, phi2) in enumerate(zip(phi1_values, phi2_values)):
        cost_values[i] = cost_positional(phi1, phi2)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(phi1_values, phi2_values, cost_values)

    plt.savefig("test-1.png", bbox_inches='tight')


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


# if __name__ == "__main__":
#     parser = ArgumentParser()
#     parser.add_argument("--phi1", default=0.012, help="Initial value for angle of rotation about x-axis")
#     parser.add_argument("--phi2", default=0.011, help="Initial value for angle of rotation about y-axis")
#     parser.add_argument("--step_size", default=0.4, help="Optimizer step size used in the optimization process.")
#     parser.add_argument("--n_steps", default=100, help="Number of optimization steps to take.")
#
#     args = parser.parse_args()
#     main(args)






