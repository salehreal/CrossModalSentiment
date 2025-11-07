import numpy as np
from sklearn.metrics import accuracy_score

def fitness(X, y, selected, model):
    if np.sum(selected) == 0:
        return 0
    X_sel = X[:, selected == 1]
    model.fit(X_sel, y)
    pred = model.predict(X_sel)
    acc = accuracy_score(y, pred)
    return acc - 0.01 * np.sum(selected)

def hgbee(X, y, model, max_iter=20, pop_size=10):
    dim = X.shape[1]
    pop = np.random.randint(0, 2, (pop_size, dim))
    best = pop[0]
    best_score = fitness(X, y, best, model)

    for _ in range(max_iter):
        for i in range(pop_size):
            new = np.copy(pop[i])
            flip = np.random.randint(0, dim)
            new[flip] = 1 - new[flip]
            score = fitness(X, y, new, model)
            if score > best_score:
                best = new
                best_score = score
    return best
