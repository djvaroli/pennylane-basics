# PennyLane Basics

My adventure into learning more about Quantum Computing and Quantum Machine Learning
with the help of the PennyLane package in Python.

## Setup
In order to run these tutorials on your local machine just follow these simple steps.
(Assuming you have Python3.6+ installed)

```bash
git clone https://github.com/djvaroli/pennylane-basics.git pennylane-basics/
```
Create a Python3 Virtual Environment
```bash
cd pennylane-basics && python3 -m venv venv/ && source venv/bin/activate
```
Install dependencies
```text
pip install -r requirements.txt
```
You should be all set to run the scripts!

## Tutorial Scripts
### Qubit Rotation
In this section I follow along with the PennyLane tutorial on Qubit rotation, which is 
also the first tutorial offered on the PennyLane website.

### Variational Quantum Eigensolver (VQE) Basics
In this part I follow along the tutorial on how to implement a VQE with pennylane.
The VQE algorithm is used in quantum chemistry, with the goal of using a quantum computer to
estimate the expectation of the molecule's electronic Hamiltonian and a classical optimizer
is used to adjust the parameters of the quantum circuit in order to find the molecules ground state energy.

That is a mouthful! I still have lot's of work to do, before I can confidently say I understand what all of that means
exactly, but I think just trying and slowly learning is the way to go!