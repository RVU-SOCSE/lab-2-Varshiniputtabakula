import numpy as np

# Data
X = np.array([
    [38, 420000, 0],
    [52, 360000, 5],
    [45, 780000, 0],
    [29, 300000, 12],
    [61, 500000, 8]
])

y = np.array([-1, 1, 1, 1, 1])  # Illness encoding
w = np.array([0.2] * 5)

def stump_smoking(x):
    return np.where(x[:,2] >= 1, 1, -1)

def stump_age(x):
    return np.where(x[:,0] >= 45, 1, -1)

def stump_income(x, threshold=600000):
    return np.where(x[:,1] < threshold, 1, -1)

def adaboost_round(h, X, y, w):
    pred = h(X)
    err = np.sum(w[pred != y])
    alpha = 0.5 * np.log((1 - err) / err)
    w = w * np.exp(-alpha * y * pred)
    w = w / np.sum(w)
    return alpha, w, err

# Round 1
a1, w, _ = adaboost_round(stump_smoking, X, y, w)

# Round 2
a2, w, _ = adaboost_round(stump_age, X, y, w)

# Round 3
a3, w, _ = adaboost_round(stump_smoking, X, y, w)

# Round 4
a4, w, _ = adaboost_round(
    lambda x: stump_income(x, 600000),
    X, y, w
)

print("Alphas:", a1, a2, a3, a4)
print("Final Weights:", w)
