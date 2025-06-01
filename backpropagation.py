import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class XORModel_Backpropagation:
    def __init__(self, num_hidden_nodes, num_epochs, learning_rate, print_interval):
        self.W1 = np.random.randn(2, num_hidden_nodes)
        self.W2 = np.random.randn(num_hidden_nodes, 1)
        self.b1 = np.random.randn(num_hidden_nodes)
        self.b2 = np.random.randn(1)
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.print_interval = print_interval

    def __call__(self, X, ytrue):
        for epoch in range(self.num_epochs):
            Z1 = X @ self.W1 + self.b1        # (4, 2)
            s1 = sigmoid(Z1)        # (4, 2)

            Z2 = s1 @ self.W2 + self.b2        # (4, 1)
            ypred = sigmoid(Z2)       # (4, 1)
            E = 0.5 * np.sum((ypred - ytrue) ** 2)

            E_PartialDerivative_ypred = ypred - ytrue       # (4, 1)
            ypred_PartialDerivative_Z2 = ypred * (1 - ypred)        # (4, 1)
            E_PartialDerivative_Z2 = E_PartialDerivative_ypred * ypred_PartialDerivative_Z2       # (4, 1)
            Z2_PartialDerivative_W2 = s1        # (4, 2)
            E_PartialDerivative_W2 = Z2_PartialDerivative_W2.T @ E_PartialDerivative_Z2        # (2, 1)
            E_PartialDerivative_b2 = np.sum(E_PartialDerivative_Z2, axis=0)       # (1)

            Z2_PartialDerivative_s1 = self.W2        # (2, 1)
            E_PartialDerivative_s1 = E_PartialDerivative_Z2 @ Z2_PartialDerivative_s1.T        # (4, 2)
            s1_PatrialDerivative_Z1 = s1 * (1 - s1)       # (4, 2)
            E_PartialDerivative_Z1 = E_PartialDerivative_s1 * s1_PatrialDerivative_Z1       # (4, 2)
            Z1_PartialDerivative_W1 = X       # (4, 2)
            E_PartialDerivative_W1 = Z1_PartialDerivative_W1.T @ E_PartialDerivative_Z1        # (2, 2)
            E_PartialDerivative_b1 = np.sum(E_PartialDerivative_Z1, axis=0)       # (1)

            Δw1 = -self.learning_rate * E_PartialDerivative_W1
            self.W1 += Δw1
            Δw2 = -self.learning_rate * E_PartialDerivative_W2
            self.W2 += Δw2
            Δb1 = -self.learning_rate * E_PartialDerivative_b1
            self.b1 += Δb1
            Δb2 = -self.learning_rate * E_PartialDerivative_b2
            self.b2 += Δb2

            if (epoch+1) % self.print_interval == 0:
                loss = 0.5 * np.sum((ypred - ytrue) ** 2)
                print(f"Epoch {epoch+1}, Loss: {loss:.6f}")
        
        print(ypred)
        return ypred
