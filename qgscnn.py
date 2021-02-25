import pennylane as qml
from matplotlib import pyplot as plt
import numpy as np
import scipy
import networkx as nx
import copy



# this should be 1 qubit precision
def coupling_hamiltonian(param1, param2, qubits, graph, trotter_step):
    # With the quantum data defined, we are able to construct the QGRNN and learn the target Hamiltonian. Each of the
    # exponentiated Hamiltonians in the QGRNN ansatz are the ZZ, Z and X terms from the Ising Hamiltonian. This gives:

    # Applies a layer of RZZ gates (based on a graph)
    for count, i in enumerate(graph.edges):
        qml.MultiRZ(2 * param1[count] * trotter_step, wires=[i[0], i[1]])

    # Applies a layer of RZ gates
    for count, i in enumerate(qubits):
        qml.RZ(2 * param2[count] * trotter_step, wires=i)

    # Applies a layer of RX gates
    for i in qubits:
        qml.RX(2 * trotter_step, wires=i)


c1 = (np.sqrt(2) - 4 * np.cos(np.pi / 8)) / (4 - 8 * np.cos(np.pi / 8))
c2 = (np.sqrt(2) - 1) / (4 * np.cos(np.pi / 8) - 2)
a = np.pi / 2
b = 3 * np.pi / 4
four_term_grad_recipe = ([[c1, 1, a], [-c1, 1, -a], [-c2, 1, b], [c2, 1, -b]],)

from pennylane.operation import AnyWires, Operation


class CouplingGate(Operation):
    num_params = 2
    num_wires = AnyWires
    par_domain = "R"
    grad_method = "A"
    grad_recipe = four_term_grad_recipe * 2

    def __init__(self, *args, **kwargs):
        # self.num_params = 2
        super().__init__(*args[1:], **kwargs)

    @classmethod
    def _matrix(cls, *params):
        U = np.asarray(params[0])

        if U.shape[0] != U.shape[1]:
            raise ValueError("Operator must be a square matrix.")

        if not np.allclose(U @ U.conj().T, np.identity(U.shape[0])):
            raise ValueError("Operator must be unitary.")

        return U

