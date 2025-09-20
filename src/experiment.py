"""The measurement that turns the leak into a number.

Three matrices are built over the SAME rows and the same honest features. They
differ only in how one extra column - the planted leaking feature - is read:

  honest    4 point-in-time-correct features.
  offline   those 4 plus `income_current_value` read with NO knowledge bound.
            This is what an analyst gets from the warehouse.
  deployed  those 4 plus the SAME feature read point-in-time-correctly. This
            is what the model will actually receive in production, because in
            production the verification has not happened yet.

The model is trained once, on `offline`, and scored on both `offline` and
`deployed` test folds. That is the honest simulation of shipping a leaking
model: the weights are whatever the leak taught them, and production then
feeds those weights the values it really has.
"""

from __future__ import annotations

import statistics

from . import generator
from .model import auc, standardise, LogisticRegression, apply_scaling
from .registry import AS_OF_LABEL


def build_matrices(dataset, rows):
    """Return (honest, offline, deployed, labels) for the given rows."""
    registry = dataset.registry
    leaking = generator.leaking_definition()
    honest_only = leaking.replace(knowledge_policy=AS_OF_LABEL)

    honest, offline, deployed, labels = [], [], [], []
    for entity, label_time, label in rows:
        base = [
            _z(registry.compute(name, entity, label_time))
            for name in generator.HONEST_FEATURES
        ]
        leak_value = _z(registry.compute_unaudited(leaking, entity, label_time))
        pit_value = _z(registry.compute_unaudited(honest_only, entity, label_time))
        honest.append(list(base))
        offline.append(base + [leak_value])
        deployed.append(base + [pit_value])
        labels.append(label)
    return honest, offline, deployed, labels


def _z(value):
    return 0.0 if value is None else value


def run_experiment(seed: int = 7, n_entities: int = 400) -> dict:
    dataset = generator.generate(seed=seed, n_entities=n_entities)
    train_rows, test_rows = dataset.split()
    h_tr, o_tr, _, y_tr = build_matrices(dataset, train_rows)
    h_te, o_te, d_te, y_te = build_matrices(dataset, test_rows)

    honest_auc = _fit_score(h_tr, y_tr, h_te, y_te)[0]

    scaled, means, sds = standardise(o_tr)
    leak_model = LogisticRegression().fit(scaled, y_tr)
    offline_auc = auc(leak_model.score(apply_scaling(o_te, means, sds)), y_te)
    # Same weights, same scaling; only the input values differ, because
    # production cannot see a restatement that has not happened yet.
    deployed_auc = auc(leak_model.score(apply_scaling(d_te, means, sds)), y_te)

    return {
        "seed": seed,
        "n": len(dataset.rows),
        "default_rate": dataset.default_rate(),
        "honest_auc": honest_auc,
        "offline_auc": offline_auc,
        "deployed_auc": deployed_auc,
        "offline_minus_deployed": offline_auc - deployed_auc,
        "offline_minus_honest": offline_auc - honest_auc,
        "dataset": dataset,
    }


def _fit_score(train, y_train, test, y_test):
    scaled, means, sds = standardise(train)
    model = LogisticRegression().fit(scaled, y_train)
    return auc(model.score(apply_scaling(test, means, sds)), y_test), model


def replicate(seeds, n_entities: int = 250) -> dict:
    """Repeat the experiment over independent seeds and return the mean and
    standard error of each headline quantity.

    The suite sets its pass bars from these standard errors rather than from a
    round number: an effect is claimed only when the replicate mean is many
    standard errors away from the null.
    """
    results = [run_experiment(seed=s, n_entities=n_entities) for s in seeds]
    summary = {"seeds": list(seeds), "n_entities": n_entities, "runs": results}
    for key in (
        "honest_auc",
        "offline_auc",
        "deployed_auc",
        "offline_minus_deployed",
        "offline_minus_honest",
    ):
        values = [r[key] for r in results]
        summary[key + "_mean"] = statistics.fmean(values)
        summary[key + "_se"] = (
            statistics.stdev(values) / len(values) ** 0.5 if len(values) > 1 else 0.0
        )
        summary[key + "_values"] = values
    return summary
