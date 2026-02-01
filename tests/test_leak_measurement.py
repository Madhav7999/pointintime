"""The planted leak, measured with a model - and the naive join shown to leak.

Pass bars are derived from replicate standard errors (5 independent seeds),
not from round numbers: an effect is asserted only when its replicate mean is
several standard errors from the null. Measured at the time of writing
(seeds 1-5, 250 entities): offline-minus-deployed AUC mean 0.225, SE 0.024.
"""

import unittest

from src import generator
from src.experiment import replicate
from src.registry import AS_OF_LABEL, LeakageError

SEEDS = (1, 2, 3, 4, 5)
# 4 SE: under a normal approximation the chance a zero effect clears this by
# noise is ~3e-5. Anything weaker would be a claim the data barely supports.
Z = 4.0


class NaiveJoinInflatesOfflineAucTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = replicate(SEEDS, n_entities=250)

    def test_naive_join_inflates_offline_auc_over_honest_model(self):
        s = self.summary
        self.assertGreater(
            s["offline_minus_honest_mean"] - Z * s["offline_minus_honest_se"], 0.0,
            "the latest-value join must visibly inflate offline AUC; if it does "
            "not, the planted leak is not a leak and the whole suite is vacuous",
        )

    def test_leaking_model_collapses_when_fed_production_values(self):
        s = self.summary
        self.assertGreater(
            s["offline_minus_deployed_mean"] - Z * s["offline_minus_deployed_se"], 0.0
        )

    def test_every_seed_shows_the_gap(self):
        # Not a threshold: just the sign, per seed. A leak that only shows up
        # on average is not the planted mechanism.
        for gap in self.summary["offline_minus_deployed_values"]:
            self.assertGreater(gap, 0.0)

    def test_deployed_auc_is_indistinguishable_from_honest_model(self):
        # The leaked feature is worthless in production, so the deployed model
        # should land near the honest one - not above it. Paired differences.
        diffs = [
            d - h for d, h in zip(self.summary["deployed_auc_values"],
                                  self.summary["honest_auc_values"])
        ]
        mean = sum(diffs) / len(diffs)
        var = sum((x - mean) ** 2 for x in diffs) / (len(diffs) - 1)
        se = (var / len(diffs)) ** 0.5
        self.assertLess(abs(mean), Z * se + 1e-9,
                        "deployed AUC differs from honest AUC beyond noise")

    def test_the_offline_leak_advantage_is_spurious_not_real_signal(self):
        # If the leak carried real decision-time signal, deployed AUC would
        # exceed the honest model. It does not (see previous test), and the
        # offline advantage exceeds any honest-vs-deployed difference.
        s = self.summary
        self.assertGreater(s["offline_auc_mean"], s["deployed_auc_mean"])
        self.assertGreater(s["offline_auc_mean"], s["honest_auc_mean"])


class NaiveLatestJoinIsWrongTests(unittest.TestCase):
    """The generic approach - join the current value - against ground truth."""

    @classmethod
    def setUpClass(cls):
        cls.data = generator.generate(seed=11, n_entities=120)

    def test_latest_join_returns_post_decision_restatement_for_every_applicant(self):
        store = self.data.store
        leaked = 0
        for entity, label_time, _ in self.data.rows:
            naive = store.unsafe_latest(entity, "annual_income", label_time).value
            pit = store.as_of(entity, "annual_income", label_time).value
            if naive != pit:
                leaked += 1
        # Ground truth: every income fact is restated exactly once, always
        # after the decision. So the naive join is wrong for ALL rows.
        self.assertEqual(self.data.truth["revisions_per_income_fact"], 2)
        self.assertEqual(leaked, len(self.data.rows))

    def test_naive_restated_income_correlates_with_the_label(self):
        # Ground truth: restatement gap = 11000 USD between classes.
        store = self.data.store
        drops = {0: [], 1: []}
        for entity, label_time, label in self.data.rows:
            naive = store.unsafe_latest(entity, "annual_income", label_time).value
            pit = store.as_of(entity, "annual_income", label_time).value
            drops[label].append(pit - naive)
        gap = (sum(drops[1]) / len(drops[1])) - (sum(drops[0]) / len(drops[0]))
        truth = self.data.truth["restatement_gap"]
        # Standard error of a difference of two means with SD 3000.
        se = self.data.truth["restatement_sd"] * (
            1 / len(drops[1]) + 1 / len(drops[0])) ** 0.5
        self.assertLess(abs(gap - truth), Z * se)

    def test_pit_read_never_uses_a_revision_known_after_the_label(self):
        store = self.data.store
        for entity, label_time, _ in self.data.rows:
            value = store.as_of(entity, "annual_income", label_time).value
            allowed = store.connection.execute(
                "SELECT value FROM facts WHERE entity_id=? AND attribute="
                "'annual_income' AND knowledge_time<=? "
                "ORDER BY knowledge_time DESC LIMIT 1",
                (entity, label_time),
            ).fetchone()[0]
            self.assertEqual(value, allowed)

    def test_late_transactions_make_backfilled_counts_exceed_pit_counts(self):
        store = self.data.store
        more = 0
        for entity, label_time, _ in self.data.rows:
            pit = store.as_of(entity, "transaction_amount", label_time, "count", 90.0)
            naive = store.unsafe_latest(entity, "transaction_amount", label_time,
                                        "count", 90.0)
            self.assertGreaterEqual(naive.value, pit.value)
            more += naive.value > pit.value
        self.assertGreater(more, 0, "planted late transactions never mattered")

    def test_store_refuses_to_put_the_leak_in_a_training_matrix(self):
        rows = self.data.rows[:3]
        with self.assertRaises(LeakageError):
            self.data.registry.training_matrix([generator.LEAKING_FEATURE], rows)
        matrix, labels = self.data.registry.training_matrix(
            list(generator.HONEST_FEATURES), rows)
        self.assertEqual(len(matrix), 3)

    def test_future_dated_event_is_excluded_from_pit_read(self):
        entity, label_time, _ = self.data.rows[0]
        from src.store import Observation
        from src.timeline import shift
        before = self.data.store.as_of(entity, "transaction_amount", label_time,
                                       "count").value
        future = shift(label_time, days=2)
        self.data.store.append_many([Observation(
            "FUTURE|1", entity, "transaction_amount", future, future, 1e6)])
        after = self.data.store.as_of(entity, "transaction_amount", label_time,
                                      "count").value
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
