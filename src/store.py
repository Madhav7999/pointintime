"""A bitemporal fact store on SQLite, with the knowledge-time predicate
enforced by the store instead of by the caller.

The design decision this file exists to make:

    The point-in-time read API has NO knowledge-time parameter.

`as_of(entity, attribute, label_time, ...)` derives the knowledge cutoff from
the label time and injects it into the SQL itself. A caller cannot forget it,
cannot widen it, and cannot pass `None` for it, because there is nowhere to
pass it. A generic feature store instead offers something like
`get_features(entities, timestamps=None)` and documents that you should pass
timestamps; a default of "now" is a leak with a docstring.

The escape hatches are deliberately ugly and deliberately named:

  * `audit_as_of(..., knowledge_time=...)` - read the store as a historian:
    "what did we believe on Tuesday?". Legitimate, but never used to build a
    training feature; it takes an explicit cutoff, so it can never be reached
    by accident.
  * `unsafe_latest(...)` - no knowledge bound. This is the leak vector, kept
    in the codebase on purpose so the leakage detector has something real to
    catch and so tests can quantify the damage.

Restatement model
-----------------
A restatement is a NEW ROW with the SAME `fact_id` and the SAME `event_time`,
a LATER `knowledge_time`, and a different value. The event time does not move:
the applicant's income in March was always whatever it was; only our belief
about it changed in May. Reading as of a knowledge time before the correction
must return the ORIGINAL WRONG VALUE. That is the property that separates a
bitemporal store from a table with a timestamp column, and it is what makes
backtests reproducible.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from .timeline import END_OF_TIME, days_between, shift, valid

AGGREGATIONS = ("last", "first", "sum", "count", "mean", "max", "min")


class StoreError(Exception):
    """Base class for refusals raised by the store."""


class FutureWindowError(StoreError):
    """Raised when a read asks for event times at or beyond the label time."""


@dataclass(frozen=True)
class Observation:
    """One (fact, belief) pair. Immutable; corrections append, never update."""

    fact_id: str
    entity_id: str
    attribute: str
    event_time: str
    knowledge_time: str
    value: float


@dataclass(frozen=True)
class Reading:
    """The result of a point-in-time read."""

    value: float | None
    # Event time of the newest underlying fact that contributed. Used by the
    # freshness SLA: a value whose newest input is 90 days stale is not a
    # current feature, it is a historical one wearing a current label.
    newest_event_time: str | None
    rows_visible: int


_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    fact_id        TEXT NOT NULL,
    entity_id      TEXT NOT NULL,
    attribute      TEXT NOT NULL,
    event_time     TEXT NOT NULL,
    knowledge_time TEXT NOT NULL,
    value          REAL NOT NULL,
    PRIMARY KEY (fact_id, knowledge_time)
);
CREATE INDEX IF NOT EXISTS facts_lookup
    ON facts (entity_id, attribute, event_time, knowledge_time);
"""

# The shared body of every read. `rn = 1` selects, per underlying fact, the
# newest revision the caller was ALLOWED to know about. Doing the revision
# pick with a window function (rather than MAX(knowledge_time) in a subquery)
# keeps the value and its knowledge time together, which a MAX() join loses
# when two revisions share a knowledge time.
_VISIBLE_CTE = """
WITH visible AS (
    SELECT fact_id, event_time, knowledge_time, value,
           ROW_NUMBER() OVER (
               PARTITION BY fact_id
               ORDER BY knowledge_time DESC, rowid DESC
           ) AS rn
    FROM facts
    WHERE entity_id = :entity
      AND attribute = :attribute
      AND event_time <= :event_hi
      AND event_time >  :event_lo
      AND knowledge_time <= :knowledge_cutoff
)
"""


class PointInTimeStore:
    """SQLite-backed bitemporal feature store."""

    def __init__(self, path: str = ":memory:") -> None:
        self.connection = sqlite3.connect(path)
        self.connection.executescript(_SCHEMA)
        # Query log: every read records the knowledge cutoff it actually ran
        # with. Reviewers can therefore see leakage in the log even when the
        # calling code looks innocent.
        self.query_log: list[dict] = []

    # ------------------------------------------------------------------ write

    def append(self, obs: Observation) -> None:
        if not (valid(obs.event_time) and valid(obs.knowledge_time)):
            raise StoreError("malformed timestamp on %r" % (obs.fact_id,))
        if obs.knowledge_time < obs.event_time:
            # Knowing a fact before it happened is not late data, it is a
            # clock bug or a leak already baked into the ingestion pipeline.
            raise StoreError(
                "knowledge_time %s precedes event_time %s for fact %s"
                % (obs.knowledge_time, obs.event_time, obs.fact_id)
            )
        self.connection.execute(
            "INSERT OR REPLACE INTO facts VALUES (?,?,?,?,?,?)",
            (
                obs.fact_id,
                obs.entity_id,
                obs.attribute,
                obs.event_time,
                obs.knowledge_time,
                float(obs.value),
            ),
        )

    def append_many(self, observations) -> None:
        for obs in observations:
            self.append(obs)
        self.connection.commit()

    def restate(self, fact_id: str, new_value: float, knowledge_time: str) -> None:
        """Correct a fact we already recorded, learned at `knowledge_time`.

        Event time is copied from the original: the world did not change, our
        belief did.
        """
        row = self.connection.execute(
            "SELECT entity_id, attribute, event_time FROM facts "
            "WHERE fact_id = ? ORDER BY knowledge_time LIMIT 1",
            (fact_id,),
        ).fetchone()
        if row is None:
            raise StoreError("cannot restate unknown fact %s" % fact_id)
        self.append(
            Observation(fact_id, row[0], row[1], row[2], knowledge_time, new_value)
        )
        self.connection.commit()

    # ------------------------------------------------------------------- read

    def as_of(
        self,
        entity_id: str,
        attribute: str,
        label_time: str,
        aggregation: str = "last",
        window_days: float | None = None,
        event_time_offset_days: float = 0.0,
    ) -> Reading:
        """Point-in-time-correct read. NOTE THE ABSENT PARAMETER.

        There is no `knowledge_time` argument. The cutoff is the label time,
        always, and it is injected into the SQL below. This signature is
        asserted by the test suite: adding a knowledge-time parameter here,
        however well-intentioned, is the bug.
        """
        if event_time_offset_days > 0:
            # Defence in depth. The registry's static detector catches this at
            # definition time; the store catches it again at read time, so a
            # definition constructed dynamically cannot slip past.
            raise FutureWindowError(
                "event-time window extends %.2f days past the label time; "
                "that window contains the future" % event_time_offset_days
            )
        return self._read(
            entity_id,
            attribute,
            label_time,
            aggregation,
            window_days,
            event_time_offset_days,
            knowledge_cutoff=label_time,
            mode="point_in_time",
        )

    def audit_as_of(
        self,
        entity_id: str,
        attribute: str,
        label_time: str,
        knowledge_time: str,
        aggregation: str = "last",
        window_days: float | None = None,
    ) -> Reading:
        """Historian's read: what did we believe at `knowledge_time`?

        Legitimate for reconciliation and for reproducing an old backtest.
        Never used to build a feature - the explicit cutoff makes that visible
        at every call site.
        """
        return self._read(
            entity_id,
            attribute,
            label_time,
            aggregation,
            window_days,
            0.0,
            knowledge_cutoff=knowledge_time,
            mode="audit",
        )

    def unsafe_latest(
        self,
        entity_id: str,
        attribute: str,
        label_time: str,
        aggregation: str = "last",
        window_days: float | None = None,
        event_time_offset_days: float = 0.0,
    ) -> Reading:
        """No knowledge bound: returns today's belief about yesterday's world.

        This is the leak. It is in the codebase on purpose, it is named after
        what it does, and every call is stamped mode='unsafe' in the query log
        so the damage can be measured rather than argued about.
        """
        return self._read(
            entity_id,
            attribute,
            label_time,
            aggregation,
            window_days,
            event_time_offset_days,
            knowledge_cutoff=END_OF_TIME,
            mode="unsafe",
        )

    # ---------------------------------------------------------------- internal

    def _read(
        self,
        entity_id: str,
        attribute: str,
        label_time: str,
        aggregation: str,
        window_days: float | None,
        event_time_offset_days: float,
        knowledge_cutoff: str,
        mode: str,
    ) -> Reading:
        if aggregation not in AGGREGATIONS:
            raise StoreError("unknown aggregation %r" % (aggregation,))
        event_hi = (
            label_time
            if not event_time_offset_days
            else shift(label_time, days=event_time_offset_days)
        )
        event_lo = "" if window_days is None else shift(event_hi, days=-window_days)
        params = {
            "entity": entity_id,
            "attribute": attribute,
            "event_hi": event_hi,
            "event_lo": event_lo,
            "knowledge_cutoff": knowledge_cutoff,
        }
        self.query_log.append(
            {"mode": mode, "attribute": attribute, "entity": entity_id, **params}
        )
        sql = _VISIBLE_CTE + (
            "SELECT event_time, value FROM visible WHERE rn = 1 "
            "ORDER BY event_time, knowledge_time, fact_id"
        )
        rows = self.connection.execute(sql, params).fetchall()
        return _aggregate(rows, aggregation)

    # ------------------------------------------------------------- inspection

    def attributes(self) -> list[str]:
        return [
            r[0]
            for r in self.connection.execute(
                "SELECT DISTINCT attribute FROM facts ORDER BY attribute"
            )
        ]

    def entities(self) -> list[str]:
        return [
            r[0]
            for r in self.connection.execute(
                "SELECT DISTINCT entity_id FROM facts ORDER BY entity_id"
            )
        ]

    def row_count(self) -> int:
        return self.connection.execute("SELECT COUNT(*) FROM facts").fetchone()[0]

    def revision_count(self, fact_id: str) -> int:
        return self.connection.execute(
            "SELECT COUNT(*) FROM facts WHERE fact_id = ?", (fact_id,)
        ).fetchone()[0]

    def ingestion_lag_days(self, fact_id: str) -> float:
        row = self.connection.execute(
            "SELECT event_time, MIN(knowledge_time) FROM facts WHERE fact_id = ?",
            (fact_id,),
        ).fetchone()
        return days_between(row[0], row[1])

    def stream(self, attribute: str | None = None):
        """All observations in knowledge-time order - i.e. the order in which
        the system actually learned things. This is the only sane replay order
        for materialising an online store; replaying in event-time order is
        exactly how a streaming pipeline reintroduces leakage."""
        sql = (
            "SELECT fact_id, entity_id, attribute, event_time, knowledge_time, value "
            "FROM facts "
            + ("WHERE attribute = :attribute " if attribute else "")
            + "ORDER BY knowledge_time, rowid"
        )
        params = {"attribute": attribute} if attribute else {}
        return [Observation(*row) for row in self.connection.execute(sql, params)]


def _aggregate(rows, aggregation: str) -> Reading:
    """Aggregate the visible revisions.

    Aggregation happens in Python only after SQL has done the revision pick,
    because the two steps compose badly: a SUM() over all matching rows would
    double-count a restated fact by adding both the wrong value and its
    correction.
    """
    if not rows:
        # count over an empty window is 0, not "unknown"; every other
        # aggregation over an empty window is genuinely unknown.
        return Reading(0.0 if aggregation == "count" else None, None, 0)
    values = [r[1] for r in rows]
    newest = rows[-1][0]
    if aggregation == "last":
        value = values[-1]
    elif aggregation == "first":
        value = values[0]
    elif aggregation == "sum":
        value = sum(values)
    elif aggregation == "count":
        value = float(len(values))
    elif aggregation == "mean":
        value = sum(values) / len(values)
    elif aggregation == "max":
        value = max(values)
    else:
        value = min(values)
    return Reading(float(value), newest, len(rows))
