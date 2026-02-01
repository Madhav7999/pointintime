"""The store's two clocks, and what a single-timestamp store gets wrong.

Every test here is written so that it would FAIL against the generic
implementation - a table with one `timestamp` column and a `WHERE timestamp <=
:as_of` predicate. That implementation passes "return the value from before
the cut-off" and fails every test about knowledge time.
"""

import inspect
import unittest

from src.store import FutureWindowError, Observation, PointInTimeStore, StoreError
from src.timeline import END_OF_TIME, ts


def obs(fact_id, event_day, knowledge_day, value, attribute="balance", entity="E1"):
    return Observation(fact_id, entity, attribute, ts(event_day), ts(knowledge_day), value)


class KnowledgeTimeTests(unittest.TestCase):
    """A value known only at 09:00 must not appear in a feature for 08:00,
    even though its event time is 07:00."""

    def setUp(self):
        self.store = PointInTimeStore()
        self.store.append_many(
            [
                Observation(
                    "F1", "E1", "balance", ts(1, hours=7), ts(1, hours=9), 500.0
                )
            ]
        )

    def test_fact_is_invisible_before_it_was_knowable(self):
        reading = self.store.as_of("E1", "balance", ts(1, hours=8))
        self.assertIsNone(
            reading.value,
            "event time 07:00 is in the past at 08:00, but the system did not "
            "learn the fact until 09:00; a one-timestamp store returns 500 here",
        )
        self.assertEqual(reading.rows_visible, 0)

    def test_fact_is_visible_once_knowable(self):
        self.assertEqual(self.store.as_of("E1", "balance", ts(1, hours=9)).value, 500.0)
        self.assertEqual(
            self.store.as_of("E1", "balance", ts(1, hours=23)).value, 500.0
        )

    def test_knowledge_time_boundary_is_inclusive(self):
        # A fact learned at exactly the label instant is knowable at that
        # instant. The alternative convention is defensible but must be a
        # decision, not an accident of >= vs >.
        self.assertEqual(self.store.as_of("E1", "balance", ts(1, hours=9)).value, 500.0)

    def test_knowing_a_fact_before_it_happened_is_rejected(self):
        with self.assertRaises(StoreError):
            self.store.append(
                Observation("F2", "E1", "balance", ts(5), ts(4), 1.0)
            )


class RestatementTests(unittest.TestCase):
    """A restated fact: same event time, later knowledge time, new value.

    This is the property that separates a bitemporal store from a timestamp
    column, and the reason an old backtest can be reproduced exactly.
    """

    def setUp(self):
        self.store = PointInTimeStore()
        self.store.append_many([obs("F1", 10, 10.5, 90000.0, "income")])
        self.store.restate("F1", 78000.0, knowledge_time=ts(40))

    def test_reading_before_the_correction_returns_the_original_wrong_value(self):
        reading = self.store.as_of("E1", "income", ts(20))
        self.assertEqual(
            reading.value,
            90000.0,
            "as of day 20 the system still believed 90000; returning the "
            "correction here is time travel",
        )

    def test_reading_after_the_correction_returns_the_correction(self):
        self.assertEqual(self.store.as_of("E1", "income", ts(50)).value, 78000.0)

    def test_correction_does_not_move_the_event_time(self):
        event_times = {
            row[0]
            for row in self.store.connection.execute(
                "SELECT DISTINCT event_time FROM facts WHERE fact_id = 'F1'"
            )
        }
        self.assertEqual(event_times, {ts(10)})
        self.assertEqual(self.store.revision_count("F1"), 2)

    def test_audit_read_reproduces_any_past_vintage(self):
        # Same label time, three different knowledge vintages.
        label = ts(60)
        self.assertEqual(
            self.store.audit_as_of("E1", "income", label, ts(20)).value, 90000.0
        )
        self.assertEqual(
            self.store.audit_as_of("E1", "income", label, ts(39)).value, 90000.0
        )
        self.assertEqual(
            self.store.audit_as_of("E1", "income", label, ts(41)).value, 78000.0
        )

    def test_unsafe_latest_sees_the_correction_from_any_label_time(self):
        # The control: the store is not simply hiding late rows from everyone.
        # The unsafe accessor returns the correction even for a label time
        # twenty days before the correction existed. That is the leak.
        self.assertEqual(
            self.store.unsafe_latest("E1", "income", ts(20)).value, 78000.0
        )

    def test_aggregate_counts_each_fact_once_despite_the_revision(self):
        # A naive `SELECT SUM(value) WHERE knowledge_time <= cutoff` adds both
        # the wrong value and its correction: 168000 instead of 78000.
        total = self.store.as_of("E1", "income", ts(50), "sum", window_days=365)
        self.assertEqual(total.value, 78000.0)
        self.assertEqual(total.rows_visible, 1)
        count = self.store.as_of("E1", "income", ts(50), "count", window_days=365)
        self.assertEqual(count.value, 1.0)


class WindowTests(unittest.TestCase):
    def setUp(self):
        self.store = PointInTimeStore()
        self.store.append_many(
            [
                obs("T%d" % day, day, day + 0.1, 10.0, "txn")
                for day in (10, 40, 70, 95, 99)
            ]
        )

    def test_window_is_half_open(self):
        # (label - window, label]. A closed lower bound double-counts the
        # boundary event when windows are chained.
        window = self.store.as_of("E1", "txn", ts(100), "count", window_days=30.0)
        self.assertEqual(window.value, 2.0)  # days 95 and 99
        on_bound = self.store.as_of("E1", "txn", ts(100), "count", window_days=90.0)
        self.assertEqual(
            on_bound.value, 4.0, "the day-10 event sits exactly on the excluded bound"
        )
        just_inside = self.store.as_of("E1", "txn", ts(100), "count", window_days=90.5)
        self.assertEqual(just_inside.value, 5.0)

    def test_unbounded_window_sees_everything_knowable(self):
        self.assertEqual(
            self.store.as_of("E1", "txn", ts(100), "count").value, 5.0
        )

    def test_count_over_an_empty_window_is_zero_not_unknown(self):
        reading = self.store.as_of("E1", "txn", ts(100), "count", window_days=0.5)
        self.assertEqual(reading.value, 0.0)
        self.assertIsNone(
            self.store.as_of("E1", "txn", ts(100), "mean", window_days=0.5).value,
            "the mean of no transactions is unknown, not zero",
        )

    def test_future_window_is_refused_by_the_store(self):
        with self.assertRaises(FutureWindowError):
            self.store.as_of(
                "E1", "txn", ts(100), "count", window_days=30.0,
                event_time_offset_days=15.0,
            )


class ApiShapeTests(unittest.TestCase):
    """The enforcement is structural: the safe read has nowhere to put a
    knowledge time, so no caller can forget or widen it."""

    def test_as_of_takes_no_knowledge_time_argument(self):
        params = inspect.signature(PointInTimeStore.as_of).parameters
        self.assertNotIn("knowledge_time", params)
        self.assertNotIn("knowledge_cutoff", params)
        self.assertNotIn("as_of_knowledge", params)

    def test_label_time_is_mandatory_and_positional(self):
        params = inspect.signature(PointInTimeStore.as_of).parameters
        self.assertIs(
            params["label_time"].default,
            inspect.Parameter.empty,
            "a default label time is a leak with a docstring",
        )

    def test_explicit_cutoff_paths_are_named_for_what_they_do(self):
        self.assertIn(
            "knowledge_time", inspect.signature(PointInTimeStore.audit_as_of).parameters
        )
        self.assertTrue(hasattr(PointInTimeStore, "unsafe_latest"))

    def test_every_read_logs_the_cutoff_it_actually_used(self):
        store = PointInTimeStore()
        store.append_many([obs("F1", 1, 1, 5.0)])
        store.as_of("E1", "balance", ts(3))
        store.unsafe_latest("E1", "balance", ts(3))
        modes = [entry["mode"] for entry in store.query_log]
        self.assertEqual(modes, ["point_in_time", "unsafe"])
        self.assertEqual(store.query_log[0]["knowledge_cutoff"], ts(3))
        self.assertEqual(store.query_log[1]["knowledge_cutoff"], END_OF_TIME)


if __name__ == "__main__":
    unittest.main()
