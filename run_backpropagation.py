from backpropagation import XORModel_Backpropagation
import numpy as np

X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
ytrue = np.array([[0],[1],[1],[0]], dtype=np.float32)

bp = XORModel_Backpropagation(num_hidden_nodes=4, num_epochs=50000, learning_rate=0.1, print_interval=5000)
output = bp(X, ytrue)
