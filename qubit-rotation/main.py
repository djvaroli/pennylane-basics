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


def main(args):
    """
    Main script function that computes circuit result and gradients
    :param args:
    :return:
    """
    circuit_value = circuit([args.phi1, args.phi2])
    d_circuit = quantum_gradient_function(circuit, argnum=0)
    print(f"Circuit result: {circuit_value}")
    print(f"Circuit gradients: {d_circuit([args.phi1, args.phi2])}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--phi1", default=0.5, help="Rotation angle about x-axis")
    parser.add_argument("--phi2", default=0.5, help="Rotation angle about y-axis")
    args = parser.parse_args()
    main(args)






