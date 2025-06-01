from genetic_algorithm import XORModel_GeneticAlgorithm
import numpy as np

X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
ytrue = np.array([[0],[1],[1],[0]], dtype=np.float32)

ga = XORModel_GeneticAlgorithm(num_organisms=10, num_genes=17, num_offsprings=20, X=X, ytrue=ytrue)
best_organism, best_loss = ga.train(generations=200, print_interval=50)
output = ga.predict(X, best_organism)
