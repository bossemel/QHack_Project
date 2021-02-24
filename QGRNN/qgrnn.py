# Source: https://pennylane.ai/qml/demos/qgrnn.html

import pennylane as qml
from matplotlib import pyplot as plt
import numpy as np
import scipy
import networkx as nx
import copy
import remote_cirq

API_KEY = ""
sim = remote_cirq.RemoteSimulator(API_KEY)


def run_vqe(use_remote=False):
    N = 15  # The number of pieces of quantum data that are used for each step
    max_time = 0.1  # The maximum value of time that can be used for quantum data

    def cost_function(params):
        # Separates the parameter list
        weight_params = params[0:6]
        bias_params = params[6:10]
        # Randomly samples times at which the QGRNN runs
        times_sampled = [np.random.uniform() * max_time for i in range(0, N)]
        # Cycles through each of the sampled times and calculates the cost
        total_cost = 0
        for i in times_sampled:
            result = qnode(weight_params, bias_params, time=i)
            total_cost += -1 * result
        return total_cost / N
    # Defines the new device
    if use_remote:
        qgrnn_dev = qml.device("cirq.simulator",
                         wires=2 * qubit_number + 1,
                         simulator=sim,
                         analytic=True)
    else:
        qgrnn_dev = qml.device("default.qubit", wires=2 * qubit_number + 1)

    # Defines the new QNode

    qnode = qml.QNode(qgrnn, qgrnn_dev)

    iterations = 0
    optimizer = qml.AdamOptimizer(stepsize=0.5)
    steps = 10
    qgrnn_params = list([np.random.randint(-20, 20)/50 for i in range(0, 10)])
    init = copy.copy(qgrnn_params)

    # Executes the optimization method

    for i in range(0, steps):
        qgrnn_params = optimizer.step(cost_function, qgrnn_params)
        if iterations % 5 == 0:
            # print(
            #     "Fidelity at Step " + str(iterations) + ": " + str((-1 * total_cost / N)._value)
            #     )
            print(" Mean Parameters at Step " + str(iterations) + ": " + str(np.mean(qgrnn_params)))
            print("---------------------------------------------")

    return qgrnn_params, init


def qgrnn(params1, params2, time=None):
    # Prepares the low energy state in the two registers
    qml.QubitStateVector(np.kron(low_energy_state, low_energy_state), wires=reg1+reg2)

    # Evolves the first qubit register with the time-evolution circuit to
    # prepare a piece of quantum data
    state_evolve(ham_matrix, reg1, time)

    # Applies the QGRNN layers to the second qubit register
    depth = time / trotter_step  # P = t/Delta
    for i in range(0, int(depth)):
        qgrnn_layer(params1, params2, reg2, new_ising_graph, trotter_step)

    # Applies the SWAP test between the registers
    swap_test(control, reg1, reg2)

    # Returns the results of the SWAP test
    return qml.expval(qml.PauliZ(control))


def swap_test(control, register1, register2):
    qml.Hadamard(wires=control)
    for i in range(0, len(register1)):
        qml.CSWAP(wires=[int(control), register1[i], register2[i]])
    qml.Hadamard(wires=control)


def qgrnn_layer(param1, param2, qubits, graph, trotter_step):
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


def create_hamiltonian_matrix(n, graph, params):
    matrix = np.zeros((2 ** n, 2 ** n))

    # Creates the interaction component of the Hamiltonian
    for count, i in enumerate(graph.edges):
        m = 1
        for j in range(0, n):
            if i[0] == j or i[1] == j:
                m = np.kron(m, qml.PauliZ.matrix)
            else:
                m = np.kron(m, np.identity(2))
        matrix += params[0][count] * m

    # Creates the bias components of the matrix
    for i in range(0, n):
        m1 = m2 = 1
        for j in range(0, n):
            if j == i:
                m1 = np.kron(m1, qml.PauliZ.matrix)
                m2 = np.kron(m2, qml.PauliX.matrix)
            else:
                m1 = np.kron(m1, np.identity(2))
                m2 = np.kron(m2, np.identity(2))
        matrix += (params[1][i] * m1 + m2)

    return matrix


def state_evolve(hamiltonian, qubits, time):
    # Evolving the low-energy state forward in time is fairly straightforward: all we have to do is multiply the
    # initial state by a time-evolution unitary. This operation can be defined as a custom gate in PennyLane:
    U = scipy.linalg.expm(-1j* hamiltonian * time)
    qml.QubitUnitary(U, wires=qubits)


def generate_quantum_data():
    # Let us use the following cyclic graph as the target interaction graph of the Ising Hamiltonian:
    ising_graph = nx.cycle_graph(qubit_number)

    print(f"Edges: {ising_graph.edges}")

    # We can then initialize the “unknown” target parameters that describe the target Hamiltonian:
    matrix_params = [[0.56, 1.24, 1.67, -0.79], [-1.44, -1.43, 1.18, -0.93]]
    # In theory, these parameters can be any value we want, provided they are reasonably small enough that the
    # QGRNN can reach them in a tractable number of optimization steps. In matrix_params, the first list
    # represents the ZZ interaction parameters and the second list represents the single-qubit Z parameters.

    # Finally, we use this information to generate the matrix form of the Ising model Hamiltonian in the computational basis:
    ham_matrix = create_hamiltonian_matrix(qubit_number, ising_graph, matrix_params)

    # The collection of quantum data needed to run the QGRNN has two components: (i) copies of a low-energy state,
    # and (ii) a collection of time-evolved states, each of which are simply the low-energy state evolved to
    #different times. The following is a low-energy state of the target Hamiltonian:
    low_energy_state = [
        (-0.054661080280306085 + 0.016713907320174026j),
        (0.12290003656489545 - 0.03758500591109822j),
        (0.3649337966440005 - 0.11158863596657455j),
        (-0.8205175732627094 + 0.25093231967092877j),
        (0.010369790825776609 - 0.0031706387262686003j),
        (-0.02331544978544721 + 0.007129899300113728j),
        (-0.06923183949694546 + 0.0211684344103713j),
        (0.15566094863283836 - 0.04760201916285508j),
        (0.014520590919500158 - 0.004441887836078486j),
        (-0.032648113364535575 + 0.009988590222879195j),
        (-0.09694382811137187 + 0.02965579457620536j),
        (0.21796861485652747 - 0.06668776658411019j),
        (-0.0027547112135013247 + 0.0008426289322652901j),
        (0.006193695872468649 - 0.0018948418969390599j),
        (0.018391279795405405 - 0.005625722994009138j),
        (-0.041350974715649635 + 0.012650711602265649j)]

    # We can verify that this is a low-energy state by numerically finding the lowest eigenvalue of the Hamiltonian
    #and comparing it to the energy expectation of this low-energy state:
    res = np.vdot(low_energy_state, (ham_matrix @ low_energy_state))
    energy_exp = np.real_if_close(res)
    print(f"Energy Expectation: {energy_exp}")
    ground_state_energy = np.real_if_close(min(np.linalg.eig(ham_matrix)[0]))
    print(f"Ground State Energy: {ground_state_energy}")

    # We have in fact found a low-energy, non-ground state, as the energy expectation is slightly greater than the
    # energy of the true ground state. This, however, is only half of the information we need. We also require a
    # collection of time-evolved, low-energy states. We don’t actually generate time-evolved quantum data quite yet,
    # but we now have all the pieces required for its preparation.
    return low_energy_state, ham_matrix, matrix_params


def plot_hamiltonian():
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(6, 6))

    axes[0].matshow(ham_matrix, vmin=-7, vmax=7, cmap='hot')
    axes[0].set_title("Target Hamiltonian", y=1.13)

    axes[1].matshow(init_ham, vmin=-7, vmax=7, cmap='hot')
    axes[1].set_title("Initial Guessed Hamiltonian", y=1.13)

    axes[2].matshow(new_ham_matrix, vmin=-7, vmax=7, cmap='hot')
    axes[2].set_title("Learned Hamiltonian", y=1.13)

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.show()


def compare_learned_target(qgrnn_params):
    # We first pick out the weights of edges (1, 3) and (2, 0)
    # and then remove them from the list of target parameters

    qgrnn_params = list(qgrnn_params)
    zero_weights = [qgrnn_params[1], qgrnn_params[4]]

    del qgrnn_params[1]
    del qgrnn_params[3]

    target_params = matrix_params[0] + matrix_params[1]

    print("Target parameters \tLearned parameters")
    for x in range(len(target_params)):
        print(f"{target_params[x]}\t\t\t{qgrnn_params[x]}")
    print(f"\nNon-Existing Edge Parameters: {zero_weights}")


def def_guessed_graph():
    # Defines the interaction graph for the new qubit system
    new_ising_graph = nx.Graph()
    new_ising_graph.add_nodes_from(range(qubit_number, 2 * qubit_number))
    new_ising_graph.add_edges_from([(4, 5), (5, 6), (6, 7), (4, 6), (7, 4), (5, 7)])
    print(f"Edges: {new_ising_graph.edges}")
    nx.draw(new_ising_graph)
    return new_ising_graph


if __name__ == '__main__':
    qubit_number = 4
    qubits = range(qubit_number)

    # In this simulation, we don’t have quantum data readily available to pass into the QGRNN, so we have to
    # generate it ourselves. To do this, we must have knowledge of the target interaction graph and the target
    # Hamiltonian.
    low_energy_state, ham_matrix, matrix_params = generate_quantum_data()

    # Defines some fixed values
    reg1 = list(range(qubit_number))  # First qubit register
    reg2 = list(range(qubit_number, 2 * qubit_number))  # Second qubit register

    control = 2 * qubit_number  # Index of control qubit
    trotter_step = 0.01  # Trotter step size

    # Initialize guessed graph as a fully connected graph. Non-existing connections will have weights approaching 0.
    new_ising_graph = def_guessed_graph()

    # Run Optimization
    qgrnn_params, init = run_vqe()

    # Create Hamiltonian using learned weights
    new_ham_matrix = create_hamiltonian_matrix(
        qubit_number, nx.complete_graph(qubit_number), [qgrnn_params[0:6], qgrnn_params[6:10]])

    # Create Hamiltonian using initial weights
    init_ham = create_hamiltonian_matrix(
        qubit_number, nx.complete_graph(qubit_number), [init[0:6], init[6:10]])

    plot_hamiltonian()

    compare_learned_target(qgrnn_params)
