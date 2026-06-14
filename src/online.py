"""Streaming materialisation to an online store, and the train/serve skew check.

The online store is built by REPLAYING the fact stream in knowledge-time order
- the order in which the system actually learned things - and stopping at a
watermark. That ordering is the whole design:

    Replaying in EVENT-TIME order is the back door through which leakage
    returns after you have fixed the offline join.

It looks harmless. Events carry a business timestamp, sorting by it produces a
tidy chronological replay, and every value in the resulting store is "as of"
the right business date. But an event that happened on Monday and arrived on
Friday is applied at Monday in that replay, so a feature materialised for
Tuesday contains data the system did not have until Friday. The offline join
is point-in-time correct, the online store is not, and the two disagree only
for the minority of entities with late data - which is exactly the minority a
sampled parity check with a loose tolerance will miss.

`materialise` does it correctly. `materialise_by_event_time` is the bug,
implemented faithfully so the test suite can measure what it costs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .registry import _or_zero
from .store import _aggregate
from .timeline import days_between


@dataclass
class OnlineStore:
    """Materialised features for low-latency serving.

    Keyed by (entity, attribute) -> {fact_id: (event_time, value)}. Keeping
    the fact_id lets a restatement REPLACE its earlier revision rather than
    accumulate beside it, which is what an append-only online cache gets
    wrong: it double-counts corrected facts in every sum and count.
    """

    watermark: str = ""
    values: dict = field(default_factory=dict)
    applied: int = 0
    buffered: int = 0

    def apply(self, obs) -> bool:
        if obs.knowledge_time > self.watermark:
            # Not yet knowable. Buffer rather than drop: it will be applied
            # when the watermark advances past its knowledge time.
            self.buffered += 1
            return False
        self.values.setdefault((obs.entity_id, obs.attribute), {})[obs.fact_id] = (
            obs.event_time,
            obs.value,
        )
        self.applied += 1
        return True

    def read(
        self,
        entity_id: str,
        attribute: str,
        as_of_event_time: str,
        aggregation: str = "last",
        window_days: float | None = None,
    ):
        from .timeline import shift

        facts = self.values.get((entity_id, attribute), {})
        lo = "" if window_days is None else shift(as_of_event_time, days=-window_days)
        rows = sorted(
            (ev, val)
            for ev, val in facts.values()
            if lo < ev <= as_of_event_time
        )
        return _aggregate(rows, aggregation)


def materialise(store, watermark: str, attributes=None) -> OnlineStore:
    """Correct materialisation: apply a fact only once it was knowable."""
    online = OnlineStore(watermark=watermark)
    for obs in store.stream():
        if attributes is None or obs.attribute in attributes:
            online.apply(obs)
    return online


def materialise_by_event_time(store, watermark: str, attributes=None) -> OnlineStore:
    """The plausible-looking bug: gate on event time instead of knowledge time.

    Kept in the codebase because a test that only exercises the correct path
    proves nothing about whether the correct path is doing anything.
    """
    online = OnlineStore(watermark=watermark)
    for obs in store.stream():
        if attributes is not None and obs.attribute not in attributes:
            continue
        if obs.event_time <= watermark:
            online.values.setdefault((obs.entity_id, obs.attribute), {})[
                obs.fact_id
            ] = (obs.event_time, obs.value)
            online.applied += 1
    return online


@dataclass
class SkewReport:
    checked: int = 0
    mismatched: int = 0
    max_absolute_difference: float = 0.0
    worst: tuple | None = None
    per_feature: dict = field(default_factory=dict)

    @property
    def mismatch_rate(self) -> float:
        return self.mismatched / self.checked if self.checked else 0.0

    def clean(self) -> bool:
        return self.mismatched == 0


def check_skew(registry, rows, feature_names, materialiser=materialise) -> SkewReport:
    """Compare the offline point-in-time value with what the online store
    would have served at the same instant.

    The watermark is set per row to the label time, because that is the moment
    the online store would have been queried. A single global watermark would
    make every historical row look skewed for an uninteresting reason.

    Tolerance is EXACT equality, not a small epsilon. Both paths compute the
    same aggregation over the same floats in the same order, so any difference
    is a difference of which facts were visible - which is the thing being
    tested. An epsilon here would hide a one-transaction disagreement, and a
    one-transaction disagreement is how this bug presents.
    """
    report = SkewReport()
    for entity, label_time, _label in rows:
        online = materialiser(registry.store, label_time)
        for name in feature_names:
            defn = registry.features[name]
            offline_value = _or_zero(registry.compute(name, entity, label_time))
            online_value = _or_zero(
                online.read(
                    entity,
                    defn.attribute,
                    label_time,
                    defn.aggregation,
                    defn.window_days,
                ).value
            )
            report.checked += 1
            difference = abs(offline_value - online_value)
            bucket = report.per_feature.setdefault(name, {"checked": 0, "skewed": 0})
            bucket["checked"] += 1
            if online_value != offline_value:
                report.mismatched += 1
                bucket["skewed"] += 1
                if difference > report.max_absolute_difference:
                    report.max_absolute_difference = difference
                    report.worst = (entity, name, offline_value, online_value)
    return report


def late_arrival_audit(store, entity_id: str, label_time: str, attribute: str):
    """How much data for this entity landed AFTER the decision was made.

    This is the number that tells you whether a backfilled training set can
    possibly match production: if 8 percent of transactions arrive after the
    decision, a naive backfill sees 8 percent more transactions than the
    scorer did.
    """
    rows = store.connection.execute(
        "SELECT event_time, knowledge_time FROM facts "
        "WHERE entity_id = ? AND attribute = ? AND event_time <= ?",
        (entity_id, attribute, label_time),
    ).fetchall()
    late = [r for r in rows if r[1] > label_time]
    return {
        "events_before_decision": len(rows),
        "arrived_after_decision": len(late),
        "max_lag_days": max(
            (days_between(r[0], r[1]) for r in rows), default=0.0
        ),
    }


# --------------------------------------------------------------------------
# Incremental materialisation: the replay a streaming job actually performs.
#
# `materialise` above rebuilds from scratch for one watermark, which is
# O(facts) per query and fine for a handful of rows. A parity check over
# 10,000 sampled entity-timestamps needs the streaming form: sort the samples
# by time, advance ONE online store's watermark monotonically, and read. The
# gate key is the only thing that differs between the correct job and the bug.
# --------------------------------------------------------------------------

KNOWLEDGE_GATE = "knowledge_time"
EVENT_GATE = "event_time"


class StreamingMaterialiser:
    def __init__(self, store, gate: str = KNOWLEDGE_GATE, attributes=None) -> None:
        if gate not in (KNOWLEDGE_GATE, EVENT_GATE):
            raise ValueError("unknown gate %r" % (gate,))
        self.gate = gate
        stream = [
            o for o in store.stream()
            if attributes is None or o.attribute in attributes
        ]
        # Stable sort: ties keep knowledge order, so restatements still land
        # after the original revision.
        self._stream = sorted(stream, key=lambda o: getattr(o, gate))
        self._cursor = 0
        self.online = OnlineStore(watermark="")

    def advance(self, watermark: str) -> int:
        if watermark < self.online.watermark:
            raise ValueError("watermarks only move forward")
        self.online.watermark = watermark
        applied = 0
        while (
            self._cursor < len(self._stream)
            and getattr(self._stream[self._cursor], self.gate) <= watermark
        ):
            obs = self._stream[self._cursor]
            self.online.values.setdefault((obs.entity_id, obs.attribute), {})[
                obs.fact_id
            ] = (obs.event_time, obs.value)
            self.online.applied += 1
            self._cursor += 1
            applied += 1
        return applied


def sample_entity_timestamps(store, n: int, lo: str, hi: str, seed: int = 0):
    """Uniformly sampled (entity, timestamp) pairs within [lo, hi]."""
    import random

    from .timeline import parse, FMT
    import datetime as _dt

    rng = random.Random(seed)
    entities = store.entities()
    start, end = parse(lo), parse(hi)
    span = (end - start).total_seconds()
    out = []
    for _ in range(n):
        moment = start + _dt.timedelta(seconds=int(rng.random() * span))
        out.append((rng.choice(entities), moment.strftime(FMT)))
    return out


def streaming_parity(registry, samples, feature_names, gate: str = KNOWLEDGE_GATE,
                     tolerance: float = 0.0) -> SkewReport:
    """Offline as-of value vs online streaming value for each sample.

    Tolerance defaults to exact equality for the reason given in `check_skew`:
    both paths aggregate the same floats in the same order, so any difference
    is a difference in which facts were visible. A tolerance parameter exists
    only so a caller comparing against a genuinely different online
    implementation (e.g. float32 Redis values) can state its own.
    """
    attributes = {registry.features[n].attribute for n in feature_names}
    job = StreamingMaterialiser(registry.store, gate=gate, attributes=attributes)
    report = SkewReport()
    for entity, moment in sorted(samples, key=lambda s: s[1]):
        job.advance(moment)
        for name in feature_names:
            defn = registry.features[name]
            offline_value = _or_zero(registry.compute(name, entity, moment))
            online_value = _or_zero(
                job.online.read(entity, defn.attribute, moment,
                                defn.aggregation, defn.window_days).value
            )
            report.checked += 1
            bucket = report.per_feature.setdefault(name, {"checked": 0, "skewed": 0})
            bucket["checked"] += 1
            difference = abs(offline_value - online_value)
            if difference > tolerance:
                report.mismatched += 1
                bucket["skewed"] += 1
                if difference > report.max_absolute_difference:
                    report.max_absolute_difference = difference
                    report.worst = (entity, moment, name, offline_value, online_value)
    return report
