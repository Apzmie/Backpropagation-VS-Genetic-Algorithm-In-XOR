import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class XORModel_GeneticAlgorithm:
    def __init__(self, num_organisms, num_genes, num_offsprings, X, ytrue):
        self.num_organisms = num_organisms
        self.num_genes = num_genes
        self.num_offsprings = num_offsprings
        self.X = X
        self.ytrue = ytrue
        self.all_organisms = [np.random.randn(num_genes) for _ in range(num_organisms)]

    def compute_loss(self, organism):       # Numbers should be modified if you change num_genes
        w1 = organism[:8].reshape(2, 4)
        b1 = organism[8:12]
        w2 = organism[12:16].reshape(4, 1)
        b2 = organism[16]

        hidden = sigmoid(self.X @ w1 + b1)
        ypred = sigmoid(hidden @ w2 + b2)
        loss = 0.5 * np.sum((ypred - self.ytrue) ** 2)
        return loss

    def evolve(self):
        organisms_list = [(organism, self.compute_loss(organism)) for organism in self.all_organisms]
        sorted_organisms = sorted(organisms_list, key=lambda x: x[1])      
        parent1 = sorted_organisms[0][0]
        parent2 = sorted_organisms[1][0]
        new_genes = (parent1 + parent2) / 2
        offsprings = []

        for _ in range(self.num_offsprings):
            mutation = np.random.normal(0, 0.1, size=self.num_genes)
            offspring = new_genes + mutation
            offsprings.append(offspring)

        offspring_scores = [(offspring, self.compute_loss(offspring)) for offspring in offsprings]
        sorted_offsprings = sorted(offspring_scores, key=lambda x: x[1])
        self.all_organisms = [offspring for (offspring, _) in sorted_offsprings[:self.num_organisms]]

        return sorted_offsprings[0][0]

    def train(self, generations, print_interval):
        best_loss = float('inf')
        best_organism = None

        for generation in range(generations):
            current_best_organism = self.evolve()
            loss = self.compute_loss(current_best_organism)

            if (generation+1) % print_interval == 0:
                print(f"Generation {generation+1}, Loss: {loss:.6f}")

            if loss < best_loss:
                best_loss = loss
                best_organism = current_best_organism

        return best_organism, best_loss

    def predict(self, X, organism):       # Numbers should be modified if you change num_genes
        w1 = organism[:8].reshape(2, 4)
        b1 = organism[8:12]
        w2 = organism[12:16].reshape(4, 1)
        b2 = organism[16]

        hidden = sigmoid(X @ w1 + b1)
        ypred = sigmoid(hidden @ w2 + b2)
        print(ypred)
        return ypred
