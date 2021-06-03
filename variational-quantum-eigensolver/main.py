from typing import *

import pennylane as qml
from pennylane import qchem
from pennylane import numpy as np

# load in the file with the molecule geometry
geometry = "h2.xyz"

# we consider a neutral molecule in this case
charge = 0

# set the multiplicty to 1, this parameter is related to the number of unpaired electrons in the Hatree-Fock state
# TODO Read about the Hartree-Fock approximiation and multiplicity
multiplicity = 1

# specifying the basis set to approximate atomic orbitals
basis_set = 'sto-3g'

symbols, coordinates = qchem.read_structure(geometry)
h, qubits = qchem.molecular_hamiltonian(
    symbols,
    coordinates,
    charge=charge,
    mult=multiplicity,
    basis=basis_set,
    active_electrons=2,
    active_orbitals=2,
    mapping="jordan_wigner"
)

print()
print('Number of qubits = ', qubits)
print('Hamiltonian is ', h)
print()

# initialize the device, a quibit simulator in this case. As the number of subsystems we use the number of qubits
# needed to represent the qubit Hamiltonian of the molecule
dev = qml.device('default.qubit', wires=qubits)


def circuit(params: Union[List, Tuple], wires: Union[List, Tuple]):
    """
    Quantum function where we apply single qubit rotations on each wire
    and then entangling CNOT gates.
    :param params:
    :param wires:
    :return:
    """

    # initialize the basis state and perform rotations on every wire
    qml.BasisState(np.array([1, 1, 0, 0], requires_grad=False), wires=wires)
    for i in wires:
        qml.Rot(*params[i], wires=i)

    # apply the entangling CNOT gates
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[2, 0])
    qml.CNOT(wires=[3, 1])


# our cost function and optimizer
cost_fn = qml.ExpvalCost(circuit, h, dev)
opt = qml.GradientDescentOptimizer(stepsize=0.4)

# set seed
np.random.seed(0)

# randomly initialize the parameters
params = np.random.normal(0, np.pi, (qubits, 3))

print()
print(f"Initial params: {params}")
print()

# perform the optimization over 200 steps with a convergence tolerance
max_iterations = 200
conv_tol = 1e-6

print("Starting optimization")
for step in range(max_iterations):
    params, prev_energy = opt.step_and_cost(cost_fn, params)
    energy = cost_fn(params)
    conv = np.abs(energy - prev_energy)

    if step % 20 == 0:
        print('Iteration = {:},  Energy = {:.8f} Ha'.format(step, energy))

    if conv <= conv_tol:
        break


print()
print('Final convergence parameter = {:.8f} Ha'.format(conv))
print('Final value of the ground-state energy = {:.8f} Ha'.format(energy))
print('Accuracy with respect to the FCI energy: {:.8f} Ha ({:.8f} kcal/mol)'.format(
    np.abs(energy - (-1.136189454088)), np.abs(energy - (-1.136189454088))*627.503
)
)
print()
print('Final circuit parameters = \n', params)



