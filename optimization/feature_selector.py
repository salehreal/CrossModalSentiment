import torch
import numpy as np
import random


def f_statistic(features, labels):
    """
    features: [N, D]
    labels: [N]
    خروجی: یک بردار [D] شامل F-score هر ویژگی
    """
    labels = labels.cpu().numpy()
    X = features.cpu().numpy()

    classes = np.unique(labels)
    if len(classes) < 2:
        return np.zeros(X.shape[1])

    f_scores = []
    for j in range(X.shape[1]):
        col = X[:, j]
        means = []
        vars_ = []
        ns = []

        for c in classes:
            group = col[labels == c]
            means.append(group.mean())
            vars_.append(group.var())
            ns.append(len(group))

        mean_total = col.mean()

        sb = sum(n * (m - mean_total) ** 2 for m, n in zip(means, ns))

        sw = sum(n * v for v, n in zip(vars_, ns))

        if sw == 0:
            f_scores.append(0)
        else:
            f_scores.append(sb / sw)

    return np.array(f_scores)


class FeatureSelectorGA:

    def __init__(self, input_dim, population_size=30, generations=20, mutation_rate=0.1, device="cpu"):
        self.input_dim = input_dim
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.device = device

        self.population = [np.random.randint(0, 2, size=self.input_dim) for _ in range(population_size)]

    def evolve(self, features, labels):
        features_np = features.cpu().numpy()
        labels_np = labels.cpu().numpy()

        for _ in range(self.generations):

            fitness_scores = []
            for mask in self.population:
                if mask.sum() == 0:
                    fitness_scores.append(0)
                    continue
                selected = features_np[:, mask == 1]
                f = f_statistic(torch.tensor(selected), labels)
                fitness_scores.append(f.mean())

            total_fit = sum(fitness_scores)
            probs = [f / total_fit if total_fit > 0 else 1 / len(fitness_scores) for f in fitness_scores]

            parents = random.choices(self.population, weights=probs, k=self.population_size)

            new_pop = []
            for i in range(0, self.population_size, 2):
                p1, p2 = parents[i], parents[(i + 1) % self.population_size]
                cut = random.randint(1, self.input_dim - 1)
                child1 = np.concatenate([p1[:cut], p2[cut:]])
                child2 = np.concatenate([p2[:cut], p1[cut:]])
                new_pop.extend([child1, child2])

            for child in new_pop:
                if random.random() < self.mutation_rate:
                    idx = random.randint(0, self.input_dim - 1)
                    child[idx] = 1 - child[idx]

            self.population = new_pop

        best_mask = self.population[np.argmax(fitness_scores)]
        return torch.tensor(best_mask, dtype=torch.float32, device=self.device)


class FeatureSelectorGWO:

    def __init__(self, input_dim, wolves=20, iterations=30, device="cpu"):
        self.input_dim = input_dim
        self.wolves = wolves
        self.iterations = iterations
        self.device = device

    def evolve(self, features, labels):
        features_np = features.cpu().numpy()
        labels_np = labels.cpu().numpy()

        population = np.random.randint(0, 2, (self.wolves, self.input_dim))

        for t in range(self.iterations):
            fitness = []
            for w in population:
                if w.sum() == 0:
                    fitness.append(0)
                    continue
                selected = features_np[:, w == 1]
                f = f_statistic(torch.tensor(selected), labels)
                fitness.append(f.mean())

            fitness = np.array(fitness)

            alpha = population[np.argmax(fitness)]
            beta  = population[np.argsort(fitness)[-2]]
            delta = population[np.argsort(fitness)[-3]]

            a = 2 - 2 * (t / self.iterations)

            new_pop = []
            for w in population:

                def update(target):
                    r1, r2 = np.random.rand(), np.random.rand()
                    A = 2 * a * r1 - a
                    C = 2 * r2
                    D = abs(C * target - w)
                    return target - A * D

                X1 = update(alpha)
                X2 = update(beta)
                X3 = update(delta)

                new_w = (X1 + X2 + X3) / 3
                new_w = (new_w > 0.5).astype(int)
                new_pop.append(new_w)

            population = np.array(new_pop)

        best = population[np.argmax(fitness)]
        return torch.tensor(best, dtype=torch.float32, device=self.device)


class FeatureSelectorHybrid:

    def __init__(self, input_dim, device="cpu"):
        self.ga = FeatureSelectorGA(input_dim, device=device)
        self.gwo = FeatureSelectorGWO(input_dim, device=device)
        self.device = device

    def evolve(self, features, labels):
        mask_ga = self.ga.evolve(features, labels)
        reduced = features * mask_ga

        mask_gwo = self.gwo.evolve(reduced, labels)
        return mask_gwo


def select_features(features, labels, method="HYBRID", input_dim=None, device="cpu", min_features=64):
    if input_dim is None:
        raise ValueError("input_dim must be provided")

    if method == "GA":
        selector = FeatureSelectorGA(input_dim, device=device)
    elif method == "GWO":
        selector = FeatureSelectorGWO(input_dim, device=device)
    elif method == "HYBRID":
        selector = FeatureSelectorHybrid(input_dim, device=device)
    else:
        raise ValueError("Unknown feature selection method")

    mask = selector.evolve(features, labels)

    if mask.sum() < min_features:
        print(f"⚠ Mask too small ({int(mask.sum())}). Expanding to {min_features}.")
        topk = torch.topk(mask, min_features).indices
        new_mask = torch.zeros_like(mask)
        new_mask[topk] = 1
        mask = new_mask

    return mask
