"""A small, deterministic logistic-regression scorer and a rank-based AUC.

Pure standard library: no numpy, so the gradient step is written out. The
model exists only as a MEASURING INSTRUMENT for leakage - it turns "this
feature knows the future" into a number a credit risk manager would react to.
It is deliberately plain, because a fancier learner would make the offline /
deployed AUC gap easier to attribute to the learner than to the leak.
"""

from __future__ import annotations

import math


def standardise(matrix):
    """Column-wise z-scoring. Returns (scaled, means, sds) so that the TEST
    fold can be scaled with the TRAIN fold's statistics - scaling with
    statistics pooled over both folds is itself a (mild) leak, and it would be
    embarrassing for this repo in particular."""
    if not matrix:
        return [], [], []
    n_cols = len(matrix[0])
    means = [sum(row[j] for row in matrix) / len(matrix) for j in range(n_cols)]
    sds = []
    for j in range(n_cols):
        var = sum((row[j] - means[j]) ** 2 for row in matrix) / max(
            1, len(matrix) - 1
        )
        # A constant column has sd 0. Dividing by 1 leaves it at 0 after
        # centring, which is exactly right: a constant carries no information
        # and must not be allowed to blow up into infinities. This case is not
        # hypothetical here - the leaking feature BECOMES constant when it is
        # evaluated point-in-time-correctly, which is the whole story.
        sds.append(math.sqrt(var) if var > 1e-12 else 1.0)
    scaled = [
        [(row[j] - means[j]) / sds[j] for j in range(n_cols)] for row in matrix
    ]
    return scaled, means, sds


def apply_scaling(matrix, means, sds):
    return [
        [(row[j] - means[j]) / sds[j] for j in range(len(means))] for row in matrix
    ]


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


class LogisticRegression:
    def __init__(self, l2: float = 1.0, iterations: int = 400, lr: float = 0.3):
        # L2 > 0 keeps the fit finite when a feature separates the classes
        # perfectly - which is precisely what the leaking feature does. With
        # no penalty the weights diverge and the AUC comparison becomes a
        # comparison of numerical overflow.
        self.l2 = l2
        self.iterations = iterations
        self.lr = lr
        self.weights: list[float] = []
        self.bias = 0.0

    def fit(self, matrix, labels):
        n, d = len(matrix), len(matrix[0])
        self.weights = [0.0] * d
        self.bias = 0.0
        for _ in range(self.iterations):
            grad_w = [0.0] * d
            grad_b = 0.0
            for row, y in zip(matrix, labels):
                p = _sigmoid(self.bias + sum(w * x for w, x in zip(self.weights, row)))
                err = p - y
                grad_b += err
                for j in range(d):
                    grad_w[j] += err * row[j]
            for j in range(d):
                grad_w[j] = grad_w[j] / n + self.l2 * self.weights[j] / n
                self.weights[j] -= self.lr * grad_w[j]
            self.bias -= self.lr * grad_b / n
        return self

    def score(self, matrix):
        return [
            self.bias + sum(w * x for w, x in zip(self.weights, row))
            for row in matrix
        ]


def auc(scores, labels) -> float:
    """Mann-Whitney U / rank AUC, with ties given average rank.

    Written from the rank identity rather than by sweeping a threshold grid,
    because a grid sweep quietly reports a slightly different number depending
    on grid resolution, and this project compares AUCs to three decimals.
    """
    pairs = sorted(zip(scores, labels))
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        average_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = average_rank
        i = j + 1
    positives = sum(1 for _, y in pairs if y == 1)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("AUC needs both classes present")
    rank_sum = sum(r for r, (_, y) in zip(ranks, pairs) if y == 1)
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def train_and_evaluate(train_rows, train_y, test_rows, test_y, l2: float = 1.0):
    """Fit on the train fold, report AUC on the held-out fold."""
    scaled, means, sds = standardise(train_rows)
    model = LogisticRegression(l2=l2).fit(scaled, train_y)
    test_scaled = apply_scaling(test_rows, means, sds)
    return auc(model.score(test_scaled), test_y), model, (means, sds)


def score_with(model, scaling, rows):
    means, sds = scaling
    return model.score(apply_scaling(rows, means, sds))
