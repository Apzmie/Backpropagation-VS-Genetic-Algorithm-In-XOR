# Backpropagation-VS-Genetic-Algorithm-In-XOR
Backpropagation has become the fundamental learning algorithm for neural networks. This project compares the performance between gradient-based backpropagation and gradient-free genetic algorithm in solving the XOR problem. They are implemented from scratch using NumPy, and the backpropagation algorithm is based on the paper [Learning representations by back-propagating errors](https://www.iro.umontreal.ca/~vincentp/ift3395/lectures/backprop_old.pdf).

# Overview
You only need run_backpropagation.py and run_genetic_algorithm.py to get started after download all .py files and required libraries.
- backpropagation.py - chain rule to calculate partial derivatives for updating weights
- genetic_algorithm.py - genetic imitation that weights are represented as genes, and a list of weights is represented as a single organism.
- run_backpropagation - XOR data and start of training by backpropagation
- run_genetic_algorithm - XOR data and start of training by genetic algorithm

# Backpropagation
Chain rule must be applied.

<img src="https://github.com/user-attachments/assets/d4ef2d70-a51e-4937-b429-80dff05ef741" width="400">

- We need to calculate ∂E/∂W1 to understand how weights affect the error function.
- Although weights do not exist in the error function, simplifying fractions gives us the answer.

# Genetic Algorithm
Random values are set for visualization.
<img src="https://github.com/user-attachments/assets/d24f8f56-1254-4fdc-8c24-c0fd0b655d77" width="800">

- Each organism (list) has several genes (values).
- Parents share their genes to make new offsprings with mutation (±).
- We want the best offspring with the best genes to solve the XOR problem.
- Each value becomes a weight of the neural network.

# run_backpropagation.py
It is better to set num_hidden_nodes to 4 for comparison because the number of hidden layers and hidden nodes in the genetic algorithm is fixed.
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
num_genes should be set to 17 because Linear(2, 4) -> Linear(4, 1) requires 17 values (8 weight1 + 4 bias1  + 4 weight2 + 1 bias2). num_organisms is the number of organisms at the beginning and num_offsprings is the number of offsprings from parents over time.
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

# Comparison
Common points
- If the number of hidden nodes is set to 2 rather than 4, results are inconsistent when performed multiple times.
- If the number of hidden nodes is set to 4, results are consistent and performances get better.

Difference
- Performance of the genetic algorithm shows better that loss decreases quickly with fewer generations and increasing num_offsprings makes loss decrease even faster. 
- Implementing backpropagation is simpler and much simpler if auto-grad system will be applied, whereas implementing the genetic algorithm is more complex.

My opinion
- If the number of hidden nodes is set to a large number such as 400, results of the backpropagation become inconsistent that can be considered to the same in the genetic algorithm.
- The genetic algorithm is better in XOR because it may not be getting stuck in local minima.
- Backpropagation will be better at more complex tasks because it adjust weights gradually through error correction, which enables to learn intricate patterens.

# Conclusion
Although the genetic algorithm performs better in XOR, backpropagation is more effective when considering all factors.
