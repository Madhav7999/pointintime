"""Synthetic credit-application data with a PLANTED leak.

The planted mechanism is the realistic one, not a toy. Every applicant states
an income on the application form. Some weeks or months later - always AFTER
the credit decision, and usually after the outcome is known - an income
verification process RESTATES that figure. The restatement is larger for
applicants who defaulted, because collections is what triggers a thorough
verification and because overstated income is itself a cause of default.

So the restated income column is a laundered label. It is in the warehouse, it
is called `annual_income` exactly like the honest column, and an analyst who
joins the customer dimension table to a historical decision table gets it
without doing anything that looks wrong. In the bitemporal store the two are
the same fact_id with two knowledge times, which is what makes the difference
mechanically visible instead of a matter of tribal memory.

Ground truth planted here, and asserted by the test suite:

  * `RESTATEMENT_MEAN_DEFAULT` vs `RESTATEMENT_MEAN_REPAID` - the size of the
    hindsight signal, in dollars.
  * every income fact has exactly 2 revisions, the second knowable only after
    the label time.
  * a known fraction of transactions arrive late (knowledge_time well after
    event_time), so point-in-time counts are genuinely lower than backfilled
    counts.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .registry import (
    AS_OF_LABEL,
    LATEST,
    AttributeSpec,
    FeatureDefinition,
    FeatureRegistry,
)
from .store import Observation, PointInTimeStore
from .timeline import shift, ts

# --- planted constants -------------------------------------------------------
# The verification adjustment. Defaulters' stated income is revised down by an
# order of magnitude more than repayers'. These two numbers ARE the leak: the
# gap between them, divided by the noise, is how many standard deviations of
# free signal the leaking feature hands the model.
RESTATEMENT_MEAN_DEFAULT = 12000.0
RESTATEMENT_MEAN_REPAID = 1000.0
RESTATEMENT_SD = 3000.0

# Verification lands 45-120 days after the decision. Anything in that range is
# after the label, which is all that matters; the spread just makes the
# knowledge-time column non-degenerate.
VERIFICATION_DELAY_RANGE = (45.0, 120.0)

# Share of transactions that arrive days late rather than overnight. Payment
# networks, disputes and batch reconciliations all do this.
LATE_EVENT_FRACTION = 0.08
LATE_EVENT_LAG_RANGE = (5.0, 20.0)
NORMAL_LAG_RANGE = (0.2, 3.5)

TRANSACTION_WINDOW_DAYS = 180.0
STATEMENT_LAG_DAYS = 1.0


@dataclass
class Dataset:
    store: PointInTimeStore
    registry: FeatureRegistry
    rows: list  # (entity_id, label_time, label)
    late_transaction_count: int = 0
    transaction_count: int = 0
    truth: dict = field(default_factory=dict)

    def split(self, fraction: float = 0.6):
        """Chronological split. A random split would let the model train on
        applications from after the ones it is tested on, which is a second,
        subtler time leak that most tutorials ship with."""
        ordered = sorted(self.rows, key=lambda r: r[1])
        cut = int(len(ordered) * fraction)
        return ordered[:cut], ordered[cut:]

    def default_rate(self) -> float:
        return sum(r[2] for r in self.rows) / len(self.rows)


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))


def generate(seed: int = 7, n_entities: int = 400) -> Dataset:
    rng = random.Random(seed)
    store = PointInTimeStore()
    observations = []
    rows = []
    late_count = 0
    txn_total = 0

    for i in range(n_entities):
        entity = "APP%05d" % i
        # Applications spread over a year, all with >=180 days of history.
        application_day = 200.0 + rng.random() * 330.0
        label_time = ts(application_day, hours=10)

        utilisation = min(0.98, max(0.02, rng.betavariate(2.0, 3.0)))
        n_txn = max(1, int(round(rng.gauss(26.0 - 14.0 * utilisation, 7.0))))
        income = math.exp(rng.gauss(10.8, 0.40))

        # Risk depends on utilisation, transaction activity and income. These
        # coefficients were chosen to put the honest model in the 0.70-0.80
        # AUC band that a real thin-file credit model occupies; a synthetic
        # dataset where the honest model is already at 0.95 would make the
        # leak look unimpressive for the wrong reason.
        z = (
            -1.35
            + 3.1 * (utilisation - 0.42)
            - 0.42 * ((n_txn - 20.0) / 10.0)
            - 0.45 * ((math.log(income) - 10.8) / 0.40)
            + rng.gauss(0.0, 0.85)
        )
        label = 1 if rng.random() < _sigmoid(z) else 0
        rows.append((entity, label_time, label))

        # -- credit utilisation: monthly statement snapshots ------------------
        for month in range(1, 7):
            snap_day = application_day - 30.0 * month
            snap_value = min(
                0.99, max(0.01, utilisation + rng.gauss(0.0, 0.04) * month)
            )
            event_time = ts(snap_day)
            observations.append(
                Observation(
                    "%s|util|%d" % (entity, month),
                    entity,
                    "credit_utilisation",
                    event_time,
                    shift(event_time, days=STATEMENT_LAG_DAYS),
                    snap_value,
                )
            )

        # -- transactions, some of them late ---------------------------------
        for t in range(n_txn):
            event_day = application_day - rng.random() * TRANSACTION_WINDOW_DAYS
            is_late = rng.random() < LATE_EVENT_FRACTION
            lag = rng.uniform(*(LATE_EVENT_LAG_RANGE if is_late else NORMAL_LAG_RANGE))
            event_time = ts(event_day)
            amount = math.exp(rng.gauss(3.9, 0.8))
            observations.append(
                Observation(
                    "%s|txn|%d" % (entity, t),
                    entity,
                    "transaction_amount",
                    event_time,
                    shift(event_time, days=lag),
                    amount,
                )
            )
            txn_total += 1
            late_count += 1 if is_late else 0

        # -- income: stated at application, restated after the outcome -------
        income_event = ts(application_day - 7.0)
        income_fact = "%s|income" % entity
        observations.append(
            Observation(
                income_fact,
                entity,
                "annual_income",
                income_event,
                shift(income_event, hours=12),
                income,
            )
        )
        adjustment = rng.gauss(
            RESTATEMENT_MEAN_DEFAULT if label else RESTATEMENT_MEAN_REPAID,
            RESTATEMENT_SD,
        )
        verification_time = ts(
            application_day + rng.uniform(*VERIFICATION_DELAY_RANGE)
        )
        observations.append(
            Observation(
                income_fact,
                entity,
                "annual_income",
                # Event time UNCHANGED: the applicant's income in March did
                # not change in June; only our belief about it did.
                income_event,
                verification_time,
                income - adjustment,
            )
        )

    store.append_many(observations)
    registry = _build_registry(store)
    return Dataset(
        store=store,
        registry=registry,
        rows=rows,
        late_transaction_count=late_count,
        transaction_count=txn_total,
        truth={
            "restatement_gap": RESTATEMENT_MEAN_DEFAULT - RESTATEMENT_MEAN_REPAID,
            "restatement_sd": RESTATEMENT_SD,
            "late_event_fraction": LATE_EVENT_FRACTION,
            "revisions_per_income_fact": 2,
        },
    )


# -- feature definitions ------------------------------------------------------

HONEST_FEATURES = (
    "utilisation_as_of",
    "txn_count_90d",
    "txn_mean_90d",
    "income_at_application",
)

#: The planted leak, by name. Registering it raises LeakageError.
LEAKING_FEATURE = "income_current_value"


def _build_registry(store: PointInTimeStore) -> FeatureRegistry:
    registry = FeatureRegistry(store)
    registry.declare(
        AttributeSpec(
            "annual_income",
            restatable=True,
            typical_ingestion_lag_days=0.5,
            owner="underwriting",
            description="Stated at application; restated by verification.",
        )
    )
    registry.declare(
        AttributeSpec(
            "credit_utilisation",
            restatable=False,
            typical_ingestion_lag_days=STATEMENT_LAG_DAYS,
            owner="bureau-ingest",
        )
    )
    registry.declare(
        AttributeSpec(
            "transaction_amount",
            restatable=False,
            typical_ingestion_lag_days=1.6,
            owner="payments",
        )
    )

    for defn in _honest_definitions():
        registry.register(defn)
    registry.register_quarantined(leaking_definition())
    return registry


def _honest_definitions():
    return (
        FeatureDefinition(
            name="utilisation_as_of",
            attribute="credit_utilisation",
            aggregation="last",
            max_staleness_days=45.0,
            owner="risk",
            description="Most recent statement utilisation known at decision.",
        ),
        FeatureDefinition(
            name="txn_count_90d",
            attribute="transaction_amount",
            aggregation="count",
            window_days=90.0,
            max_staleness_days=30.0,
            owner="risk",
        ),
        FeatureDefinition(
            name="txn_mean_90d",
            attribute="transaction_amount",
            aggregation="mean",
            window_days=90.0,
            max_staleness_days=30.0,
            owner="risk",
        ),
        # THE NEGATIVE CONTROL. It reads `annual_income`, the same restatable
        # attribute the leaking feature reads, and a column-name or
        # attribute-level blocklist would flag it. It is correct, because it
        # is bounded by the label time: it returns what the applicant told us
        # on the form, which is exactly what production sees.
        FeatureDefinition(
            name="income_at_application",
            attribute="annual_income",
            aggregation="last",
            knowledge_policy=AS_OF_LABEL,
            max_staleness_days=60.0,
            owner="underwriting",
            description="Stated income as known at decision time.",
        ),
    )


def leaking_definition() -> FeatureDefinition:
    """The planted leak. Same attribute and same aggregation as the negative
    control; the ONLY difference is the knowledge policy."""
    return FeatureDefinition(
        name=LEAKING_FEATURE,
        attribute="annual_income",
        aggregation="last",
        knowledge_policy=LATEST,
        owner="analytics",
        description="Current best-known income for the applicant.",
    )


def future_window_definition() -> FeatureDefinition:
    """The off-by-one leak: correct knowledge policy, window straddling the
    decision. Included so the detector is tested on more than one mechanism."""
    return FeatureDefinition(
        name="txn_count_30d_centred",
        attribute="transaction_amount",
        aggregation="count",
        window_days=30.0,
        knowledge_policy=AS_OF_LABEL,
        event_time_offset_days=15.0,
        owner="analytics",
    )
