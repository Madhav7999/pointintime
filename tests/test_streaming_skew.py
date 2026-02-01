"""Online materialisation, train/serve skew, late events, freshness SLA."""

import unittest

from src import generator
from src.online import (
    EVENT_GATE,
    KNOWLEDGE_GATE,
    StreamingMaterialiser,
    sample_entity_timestamps,
    streaming_parity,
)
from src.registry import FeatureDefinition, FeatureRegistry, StalenessError, AttributeSpec
from src.store import Observation, PointInTimeStore
from src.timeline import shift, ts

N_SAMPLES = 10000  # the README's parity sample size


class ParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.names = list(generator.HONEST_FEATURES)
        good = generator.generate(seed=7, n_entities=400)
        samples = sample_entity_timestamps(good.store, N_SAMPLES, ts(30), ts(530), 1)
        cls.good = streaming_parity(good.registry, samples, cls.names, KNOWLEDGE_GATE)
        bad = generator.generate(seed=7, n_entities=400)
        cls.bad = streaming_parity(bad.registry, samples, cls.names, EVENT_GATE)

    def test_offline_online_parity_is_exact_over_10k_samples(self):
        self.assertEqual(self.good.checked, N_SAMPLES * len(self.names))
        self.assertEqual(self.good.mismatched, 0, self.good.worst)

    def test_event_time_gated_stream_is_caught_by_the_skew_check(self):
        self.assertGreater(self.bad.mismatched, 0)

    def test_skew_is_attributed_to_the_features_with_late_or_restated_facts(self):
        pf = self.bad.per_feature
        self.assertGreater(pf["txn_count_90d"]["skewed"], 0)  # late transactions
        self.assertGreater(pf["income_at_application"]["skewed"], 0)  # restatement


class LateEventsDoNotRewriteHistoryTests(unittest.TestCase):
    """Serve a value at T; then an event dated before T arrives after T.
    Re-reading as of T must return exactly what was served."""

    def setUp(self):
        self.store = PointInTimeStore()
        self.store.append_many([
            Observation("T1", "E1", "txn", ts(10), ts(10.5), 100.0),
            Observation("T2", "E1", "txn", ts(12), ts(12.5), 50.0),
        ])
        self.t = ts(15)
        self.served_offline = self.store.as_of("E1", "txn", self.t, "sum", 30.0).value
        self.job = StreamingMaterialiser(self.store, KNOWLEDGE_GATE)
        self.job.advance(self.t)
        self.served_online = self.job.online.read("E1", "txn", self.t, "sum", 30.0).value
        # Late: happened day 14, learned day 20. Also a restatement of T1.
        self.store.append_many([Observation("T3", "E1", "txn", ts(14), ts(20), 999.0)])
        self.store.restate("T1", 1.0, knowledge_time=ts(21))

    def test_served_values_match_before_late_arrival(self):
        self.assertEqual(self.served_offline, 150.0)
        self.assertEqual(self.served_online, 150.0)

    def test_offline_replay_of_served_instant_is_unchanged(self):
        self.assertEqual(self.store.as_of("E1", "txn", self.t, "sum", 30.0).value,
                         self.served_offline)

    def test_online_rebuild_at_served_watermark_is_unchanged(self):
        job = StreamingMaterialiser(self.store, KNOWLEDGE_GATE)
        job.advance(self.t)
        self.assertEqual(job.online.read("E1", "txn", self.t, "sum", 30.0).value,
                         self.served_online)

    def test_event_time_gated_rebuild_rewrites_history(self):
        # The back door: the broken job retroactively changes the value for T.
        job = StreamingMaterialiser(self.store, EVENT_GATE)
        job.advance(self.t)
        self.assertNotEqual(job.online.read("E1", "txn", self.t, "sum", 30.0).value,
                            self.served_online)

    def test_naive_latest_join_rewrites_history(self):
        self.assertNotEqual(
            self.store.unsafe_latest("E1", "txn", self.t, "sum", 30.0).value,
            self.served_offline)

    def test_watermark_cannot_move_backwards(self):
        with self.assertRaises(ValueError):
            self.job.advance(ts(1))


class FreshnessSlaTests(unittest.TestCase):
    def setUp(self):
        self.store = PointInTimeStore()
        self.store.append_many([Observation("U1", "E1", "util", ts(10), ts(11), 0.4)])
        self.registry = FeatureRegistry(self.store)
        self.registry.declare(AttributeSpec("util", typical_ingestion_lag_days=1.0))
        self.registry.register(FeatureDefinition(
            name="util_last", attribute="util", max_staleness_days=30.0))

    def test_fresh_value_is_served(self):
        self.assertEqual(self.registry.serve("util_last", "E1", ts(30)), 0.4)

    def test_stale_value_blocks_serving(self):
        with self.assertRaises(StalenessError):
            self.registry.serve("util_last", "E1", ts(41))
        # Offline retrieval still works: staleness is a serving gate only.
        self.assertEqual(self.registry.compute("util_last", "E1", ts(41)), 0.4)

    def test_sla_boundary_is_inclusive(self):
        self.assertEqual(self.registry.serve("util_last", "E1", shift(ts(10), days=30)), 0.4)


if __name__ == "__main__":
    unittest.main()
