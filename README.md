# Backpropagation-VS-Genetic-Algorithm-In-XOR
Backpropagation has become the fundamental learning algorithm for neural networks. This project compares the performance between gradient-based backpropagation and gradient-free genetic algorithm in solving the XOR problem. They are implemented from scratch using NumPy, and the backpropagation algorithm is based on the paper [Learning representations by back-propagating errors](https://www.iro.umontreal.ca/~vincentp/ift3395/lectures/backprop_old.pdf).

# Overview
- backpropagation.py - chain rule to calculate partial derivatives for updating weights
- genetic_algorithm.py - genetic imitation that a list of weights is represented as one gene, evloving toward the best gene over time.
- run_backpropagation - XOR data and start of training by backpropagation
- run_genetic_algorithm - XOR data and start of training by genetic algorithm
