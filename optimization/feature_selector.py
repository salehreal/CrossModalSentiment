import torch
import numpy as np
import random

class FeatureSelectorGA:

    def __init__(self, input_dim, population_size=20, generations=10, mutation_rate=0.1, device="cpu"):
        self.input_dim = input_dim
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.device = device

        self.population = [self._random_mask() for _ in range(population_size)]

    def _random_mask(self):
        return np.random.choice([0, 1], size=self.input_dim)

    def _fitness(self, mask, features):
        selected = features[:, mask == 1]
        if selected.shape[1] == 0:
            return 0.0
        return np.mean(np.var(selected, axis=0))

    def evolve(self, features):
        features = features.detach().cpu().numpy()
        for _ in range(self.generations):

            fitness_scores = [self._fitness(mask, features) for mask in self.population]

            total_fit = sum(fitness_scores)
            probs = [f / total_fit if total_fit > 0 else 1.0 / len(fitness_scores) for f in fitness_scores]
            parents = random.choices(self.population, weights=probs, k=self.population_size)

            new_pop = []
            for i in range(0, self.population_size, 2):
                p1, p2 = parents[i], parents[(i+1) % self.population_size]
                cut = random.randint(1, self.input_dim - 1)
                child1 = np.concatenate([p1[:cut], p2[cut:]])
                child2 = np.concatenate([p2[:cut], p1[cut:]])
                new_pop.extend([child1, child2])

            for child in new_pop:
                if random.random() < self.mutation_rate:
                    idx = random.randint(0, self.input_dim - 1)
                    child[idx] = 1 - child[idx]

            self.population = new_pop

        fitness_scores = [self._fitness(mask, features) for mask in self.population]
        best_mask = self.population[int(np.argmax(fitness_scores))]
        return torch.tensor(best_mask, dtype=torch.float32, device=self.device)

    def apply(self, features):
        mask = self.evolve(features)
        return features * mask
