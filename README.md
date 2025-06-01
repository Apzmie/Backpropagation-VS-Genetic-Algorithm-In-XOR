# Backpropagation-VS-Genetic-Algorithm-In-XOR
Backpropagation has become the fundamental learning algorithm for neural networks. This project compares the performance between gradient-based backpropagation and gradient-free genetic algorithm in solving the XOR problem. They are implemented from scratch using NumPy, and the backpropagation algorithm is based on the paper [Learning representations by back-propagating errors](https://www.iro.umontreal.ca/~vincentp/ift3395/lectures/backprop_old.pdf).

# Overview
You only need run_backpropagation.py and run_genetic_algorithm.py to get started after download all .py files and required libraries.
- backpropagation.py - chain rule to calculate partial derivatives for updating weights
- genetic_algorithm.py - genetic imitation that a list of weights is represented as one gene, getting the gene of the best offspring
- run_backpropagation - XOR data and start of training by backpropagation
- run_genetic_algorithm - XOR data and start of training by genetic algorithm

# Genetic Algorithm
Random values are set for visualization.
<img src="https://github.com/user-attachments/assets/d24f8f56-1254-4fdc-8c24-c0fd0b655d77" width="800">

- Each organism has only one gene (list) that contains several nucleotides (weights).
- Parents share their genes to make new gene with mutation (±).
- We want the best offspring with the best gene to solve the XOR problem.
- Each value becomes a weight of the neural network.

# run_backpropagation.py
```python
from backpropagation import XORModel_Backpropagation
import numpy as np

X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
ytrue = np.array([[0],[1],[1],[0]], dtype=np.float32)

bp = XORModel_Backpropagation(num_hidden_nodes=4, num_epochs=50000, learning_rate=0.1, print_interval=5000)
output = bp(X, ytrue)
```
```text
Epoch 40000, Loss: 0.000529
Epoch 45000, Loss: 0.000461
Epoch 50000, Loss: 0.000409
[[0.01183486]
 [0.98469568]
 [0.98738753]
 [0.01684798]]
```

# run_genetic_algorithm.py
```python
from genetic_algorithm import XORModel_GeneticAlgorithm
import numpy as np

X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
ytrue = np.array([[0],[1],[1],[0]], dtype=np.float32)

ga = XORModel_GeneticAlgorithm(num_organisms=10, num_genes=17, num_offsprings=20, X=X, ytrue=ytrue)
best_organism, best_loss = ga.train(generations=200, print_interval=50)
output = ga.predict(X, best_organism)
```
```text
Generation 100, Loss: 0.029193
Generation 150, Loss: 0.000611
Generation 200, Loss: 0.000003
[[0.00136208]
 [0.99905024]
 [0.99918807]
 [0.00144575]]
```
