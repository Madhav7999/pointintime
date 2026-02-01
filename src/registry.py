"""Feature definitions, the attribute catalogue, and the serving gate.

A feature here is DATA, not a lambda. `FeatureDefinition` names the attribute
it touches, the aggregation, the event-time window, and - the field that
matters - the KNOWLEDGE POLICY: which belief-vintage of the underlying facts
the feature is allowed to see.

Making the definition declarative is what allows the leakage detector in
`leakage.py` to reason about what a feature *would* read before anyone runs
it. A feature store whose definitions are arbitrary Python functions can only
ever detect leakage by noticing that the numbers came out too good, which is
a heuristic applied after the damage.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .store import PointInTimeStore, Reading
from .timeline import days_between

# Knowledge policies. Exactly one of these is safe.
AS_OF_LABEL = "as_of_label"  # cutoff = label time. The only correct choice.
LATEST = "latest"  # cutoff = now. Reads beliefs from after the label.
PINNED_SNAPSHOT = "pinned_snapshot"  # cutoff = a fixed date. Safe only by luck.

KNOWLEDGE_POLICIES = (AS_OF_LABEL, LATEST, PINNED_SNAPSHOT)


class LeakageError(Exception):
    """Raised when the store refuses to serve a definition."""

    def __init__(self, message: str, findings=()) -> None:
        super().__init__(message)
        self.findings = list(findings)


class StalenessError(Exception):
    """Raised when a served value violates its freshness SLA."""


@dataclass(frozen=True)
class AttributeSpec:
    """What the owner of a source table tells the store about an attribute.

    `restatable` is the single most valuable field here and the one nobody
    fills in: it says this attribute's values get corrected after the fact.
    Any feature reading a restatable attribute without a knowledge bound is
    reading the correction, and the correction is frequently a function of
    the outcome you are trying to predict.
    """

    name: str
    restatable: bool = False
    typical_ingestion_lag_days: float = 0.0
    owner: str = "unowned"
    description: str = ""


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    attribute: str
    aggregation: str = "last"
    window_days: float | None = None
    knowledge_policy: str = AS_OF_LABEL
    # Positive values push the event-time window past the label time. Nobody
    # writes this on purpose; it appears as an off-by-one on a window bound
    # ("last 30 days" implemented as label +/- 15) and it is pure future.
    event_time_offset_days: float = 0.0
    pinned_knowledge_time: str | None = None
    # Freshness SLA for online serving. None = not served online.
    max_staleness_days: float | None = None
    owner: str = "unowned"
    description: str = ""
    tags: tuple = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.knowledge_policy not in KNOWLEDGE_POLICIES:
            raise ValueError("unknown knowledge policy %r" % (self.knowledge_policy,))
        if self.knowledge_policy == PINNED_SNAPSHOT and not self.pinned_knowledge_time:
            raise ValueError("pinned_snapshot policy needs pinned_knowledge_time")

    def replace(self, **changes) -> "FeatureDefinition":
        from dataclasses import replace as _replace

        return _replace(self, **changes)


class FeatureRegistry:
    """Holds attribute specs and feature definitions, and gates registration.

    Registration is the enforcement point, not retrieval. By the time someone
    is pulling training data it is too late to have an opinion; the definition
    is already in a notebook that produced a promising number.
    """

    def __init__(self, store: PointInTimeStore) -> None:
        self.store = store
        self.attributes: dict[str, AttributeSpec] = {}
        self.features: dict[str, FeatureDefinition] = {}
        self.quarantine: dict[str, list] = {}

    def declare(self, spec: AttributeSpec) -> None:
        self.attributes[spec.name] = spec

    def register(self, defn: FeatureDefinition) -> list:
        """Register a feature, refusing anything with a critical finding."""
        from .leakage import audit_definition

        findings = audit_definition(defn, self.attributes)
        blocking = [f for f in findings if f.severity == "critical"]
        if blocking:
            self.quarantine[defn.name] = findings
            raise LeakageError(
                "refusing to register %r: %s"
                % (defn.name, "; ".join(f.rule for f in blocking)),
                findings,
            )
        self.features[defn.name] = defn
        return findings

    def register_quarantined(self, defn: FeatureDefinition) -> list:
        """Record a definition that FAILED the audit, without serving it.

        Needed so the demo and the test suite can measure what the leaking
        feature would have done. Quarantined features are never returned by
        `training_matrix`; you have to ask for them by name through
        `compute_unaudited`, which is the point.
        """
        from .leakage import audit_definition

        findings = audit_definition(defn, self.attributes)
        self.quarantine[defn.name] = findings
        return findings

    # -------------------------------------------------------------- retrieval

    def compute(self, name: str, entity_id: str, label_time: str) -> float | None:
        if name not in self.features:
            raise LeakageError(
                "%r is not a registered feature (quarantined: %s)"
                % (name, name in self.quarantine)
            )
        return self._evaluate(self.features[name], entity_id, label_time).value

    def compute_unaudited(
        self, defn: FeatureDefinition, entity_id: str, label_time: str
    ) -> float | None:
        """Evaluate a definition that the audit rejected. Used only to
        quantify the damage."""
        return self._evaluate(defn, entity_id, label_time).value

    def _evaluate(
        self, defn: FeatureDefinition, entity_id: str, label_time: str
    ) -> Reading:
        if defn.knowledge_policy == AS_OF_LABEL:
            return self.store.as_of(
                entity_id,
                defn.attribute,
                label_time,
                defn.aggregation,
                defn.window_days,
                defn.event_time_offset_days,
            )
        if defn.knowledge_policy == PINNED_SNAPSHOT:
            return self.store.audit_as_of(
                entity_id,
                defn.attribute,
                label_time,
                defn.pinned_knowledge_time,
                defn.aggregation,
                defn.window_days,
            )
        return self.store.unsafe_latest(
            entity_id,
            defn.attribute,
            label_time,
            defn.aggregation,
            defn.window_days,
            defn.event_time_offset_days,
        )

    def serve(self, name: str, entity_id: str, label_time: str) -> float | None:
        """Online serving path, with the freshness SLA enforced.

        A stale feature served as current is the online mirror image of
        leakage: training saw a value computed from fresh data, production
        sees one computed from data three weeks old. Same skew, opposite
        direction of time.
        """
        defn = self.features.get(name)
        if defn is None:
            raise LeakageError("%r is not registered; refusing to serve" % name)
        reading = self._evaluate(defn, entity_id, label_time)
        if defn.max_staleness_days is not None and reading.newest_event_time:
            age = days_between(reading.newest_event_time, label_time)
            if age > defn.max_staleness_days:
                raise StalenessError(
                    "%s for %s is %.1f days stale (SLA %.1f)"
                    % (name, entity_id, age, defn.max_staleness_days)
                )
        return reading.value

    def training_matrix(self, names, rows):
        """Point-in-time-correct training data.

        `rows` is a sequence of (entity_id, label_time, label). The label time
        is a positional argument, not a keyword with a default; there is no
        way to call this without one.
        """
        matrix, labels = [], []
        for entity_id, label_time, label in rows:
            matrix.append(
                [_or_zero(self.compute(n, entity_id, label_time)) for n in names]
            )
            labels.append(label)
        return matrix, labels

    def unaudited_matrix(self, defns, rows):
        matrix, labels = [], []
        for entity_id, label_time, label in rows:
            matrix.append(
                [
                    _or_zero(self.compute_unaudited(d, entity_id, label_time))
                    for d in defns
                ]
            )
            labels.append(label)
        return matrix, labels


def _or_zero(value: float | None) -> float:
    # Missing-at-decision-time is a real state and zero is a poor encoding of
    # it, but every alternative (mean imputation from the full dataset) is
    # itself a leak. Zero after standardisation is chosen because it is the
    # only imputation that cannot import information from outside the window.
    return 0.0 if value is None else value
