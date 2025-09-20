"""Time handling for the bitemporal store.

Two clocks, never one. Every fact in this system carries BOTH:

  event_time     - when the fact became true in the world
  knowledge_time - when this system first could have known it

A store with a single timestamp column silently conflates them, and that
conflation is the entire leakage bug this repo exists to prevent. Timestamps
are stored as ISO-8601 strings with a fixed width so that lexicographic
comparison in SQL is identical to chronological comparison; this lets the
as-of predicates run as plain SQL `<=` without a UDF, which matters because
the whole point is that the predicate is enforced *inside the query*.
"""

from __future__ import annotations

import datetime as _dt

FMT = "%Y-%m-%dT%H:%M:%S"

# Every synthetic timeline in this project is anchored here so that demo
# output and test fixtures are stable and human-readable.
EPOCH = _dt.datetime(2023, 1, 1, 0, 0, 0)

# Lexicographically greater than any timestamp this system can produce.
# Used as the knowledge cutoff by the deliberately-unsafe accessor, so that
# "no knowledge bound at all" is still expressed as a bound in SQL rather
# than as a missing WHERE clause. A missing clause is invisible in review;
# a sentinel value shows up in the query log.
END_OF_TIME = "9999-12-31T23:59:59"


def ts(days: float = 0.0, hours: float = 0.0, minutes: float = 0.0) -> str:
    """Timestamp `days`/`hours`/`minutes` after the fixture epoch."""
    moment = EPOCH + _dt.timedelta(days=days, hours=hours, minutes=minutes)
    return moment.strftime(FMT)


def parse(stamp: str) -> _dt.datetime:
    return _dt.datetime.strptime(stamp, FMT)


def shift(stamp: str, days: float = 0.0, hours: float = 0.0) -> str:
    return (parse(stamp) + _dt.timedelta(days=days, hours=hours)).strftime(FMT)


def days_between(earlier: str, later: str) -> float:
    return (parse(later) - parse(earlier)).total_seconds() / 86400.0


def valid(stamp: str) -> bool:
    if stamp == END_OF_TIME:
        return True
    try:
        parse(stamp)
    except (ValueError, TypeError):
        return False
    return True
