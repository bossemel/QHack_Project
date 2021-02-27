# Quantum Spectral Graph Convolutional Neural Networks

Our project for the QHack Open Hackathon by team 'QUACKQUACKQUACK'.

### Project Description:

Over recent years, a large influx of interest has been observed in classical machine learning regarding the research into and usage of Graph Neural Networks (GNN). Part of the reason for this interest is due to their innate ability to model vast physical phenomena through the medium of pair-wise interactions between the elements of the systems. Similarly , interest in Quantum Machine Learning models is also increasing, as such architectures can leverage the computational efficiency of quantum computers and offer problem tailored solutions by [handcrafting antsatze guided by physical interactions](https://arxiv.org/abs/2006.11287). Consequently, we believe that combining these separate ideas will offer mutual benefits and improve model performance and advanced research in both fields. Seeing how GNNs are used to solve combinatorial tasks [Combinatorial optimisation and reasoning with graph neural networks by Cappart et al](https://arxiv.org/abs/2102.09544) included in workshops such as [“Deep Learning and Combinatorial Optimisation”](https://www.ipam.ucla.edu/programs/workshops/deep-learning-and-combinatorial-optimization/) help at IPAM UCLA., we would argue that it is the right time to start thinking more about Quantum Graph Neural Networks (QGNN).

We propose to implement Quantum Spectral Graph Convolutional Neural Networks (QSGCNN) as described in [Verdon et al.](https://arxiv.org/abs/1909.12264). We are planning to use the [Pennylane documentation on Quantum Graph Recurrent Neural Networks (QGRNNs)](https://pennylane.ai/qml/demos/qgrnn.html) as a guideline, and we will replace the RNN layer with a spectral convolutional layer. In particular, we want to perform unsupervised graph clustering as described in [Verdon et al.](https://arxiv.org/abs/1909.12264). We specifically want to compare the performance and inference speed between classical GNN models and their quantum counterparts on simple datasets, such as the one in [Verdon et al.](https://arxiv.org/abs/1909.12264) or k-core distilled popular GNN benchmark datasets (e.g. Cora, or Citeseer). This would primarily include the most popular and basic models based on the SGCNNs and as a stretch goal also on GraphSAGE. The results would be then compared with standard graph partitioning algorithms.

### Achievements:

- set up a target hamiltonian
- set up a qgcnn layer
- set up a training function to find the low energy states of the hamiltonian

### Future work:

- use the predicted weights to find the eigenvectors and eigenvalues of the Laplacian matrix and then generate cluster predictions from there
- change the ansatz for better training: QAOA ansatz or ArbitraryUnitary gates
- train our model on a bigger input graph
- set up a comparison to classical models

## Installation

Create a local environment and install the requirements:
```
python3 -m venv .env
source .env/bin/activate
pip3 install -r requirements.txt
```
