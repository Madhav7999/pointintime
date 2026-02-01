"""The leakage detector: static analysis over feature DEFINITIONS.

What this detector does NOT do is look at model accuracy. Accuracy-based
leakage detection ("this feature has suspiciously high AUC, flag it") is a
heuristic with two failure modes that make it unfit as a gate:

  * False negatives. A leak worth 0.03 AUC on a model that is genuinely good
    is invisible against the noise, and 0.03 AUC is a large amount of money
    in credit decisioning.
  * False positives. Genuinely strong features exist. Prior default count is
    an enormous predictor of default and is entirely legitimate. A gate that
    flags whatever predicts well trains its users to click "ignore", and a
    gate whose alerts are routinely ignored is not a gate.

The rules below reason instead about which attribute a definition touches and
under which time predicates it touches it. That answers a decidable question -
"could this value have been known at the label time?" - rather than an
undecidable one - "is this too good to be true?".

The cost of being definitional: the detector is only as good as the attribute
catalogue. An attribute whose owner did not mark it `restatable` gets the
weaker generic check. That limitation is stated in the README rather than
hidden here.
"""

from __future__ import annotations

from dataclasses import dataclass

from .registry import AS_OF_LABEL, LATEST, PINNED_SNAPSHOT, AttributeSpec

CRITICAL = "critical"
WARNING = "warning"


@dataclass(frozen=True)
class Finding:
    feature: str
    rule: str
    severity: str
    message: str

    def __str__(self) -> str:
        return "[%s] %s: %s" % (self.severity.upper(), self.rule, self.message)


def audit_definition(defn, attributes: dict) -> list:
    """Return every finding for one definition. Order is rule order, so the
    output is stable enough to assert on."""
    declared = defn.attribute in attributes
    spec = attributes.get(defn.attribute) or AttributeSpec(defn.attribute)
    findings = []
    for rule in _RULES:
        finding = rule(defn, spec, declared)
        if finding is not None:
            findings.append(finding)
    return findings


def audit_registry(registry) -> dict:
    return {
        name: audit_definition(defn, registry.attributes)
        for name, defn in registry.features.items()
    }


def is_leaking(defn, attributes: dict) -> bool:
    return any(f.severity == CRITICAL for f in audit_definition(defn, attributes))


# ------------------------------------------------------------------- the rules


def _rule_knowledge_time_unbounded(defn, spec, declared):
    """The core rule. A feature computed for label time T may only read facts
    the system knew by T. `latest` reads whatever we believe now, which for a
    training row labelled eight months ago includes eight months of hindsight.
    """
    if defn.knowledge_policy == LATEST:
        return Finding(
            defn.name,
            "KNOWLEDGE_TIME_UNBOUNDED",
            CRITICAL,
            "knowledge_policy=%r reads facts by their current value, not the "
            "value known at the label time; any correction applied after the "
            "label is visible to training and not to production"
            % (defn.knowledge_policy,),
        )
    return None


def _rule_restated_attribute_unbounded(defn, spec, declared):
    """Why the previous rule is fatal rather than merely sloppy, when the
    attribute is one that gets corrected.

    The realistic mechanism: a verification or collections process revisits an
    applicant's figures AFTER the outcome is known, and the revision is a
    function of the outcome. The restated column is then a laundered label.
    """
    if spec.restatable and defn.knowledge_policy != AS_OF_LABEL:
        return Finding(
            defn.name,
            "RESTATED_ATTRIBUTE_WITHOUT_PIT_BOUND",
            CRITICAL,
            "attribute %r is declared restatable and is read under policy %r; "
            "restatements are frequently caused by the outcome being "
            "predicted, so the corrected value is a proxy for the label"
            % (defn.attribute, defn.knowledge_policy),
        )
    return None


def _rule_future_event_window(defn, spec, declared):
    """An event-time window whose upper bound sits past the label time. This
    is the off-by-one version of the bug: the knowledge policy is correct and
    the window still contains the future."""
    if defn.event_time_offset_days > 0:
        return Finding(
            defn.name,
            "FUTURE_EVENT_WINDOW",
            CRITICAL,
            "event-time window ends %.2f days AFTER the label time; the "
            "window includes events that had not happened yet"
            % defn.event_time_offset_days,
        )
    return None


def _rule_pinned_snapshot(defn, spec, declared):
    """A pinned knowledge time is correct for exactly one label time and wrong
    for every other. It is common in backfill scripts, where somebody pinned
    the snapshot to make an old backtest reproducible and then left it in."""
    if defn.knowledge_policy == PINNED_SNAPSHOT:
        return Finding(
            defn.name,
            "PINNED_KNOWLEDGE_SNAPSHOT",
            CRITICAL,
            "knowledge time is pinned to %s for every row; rows labelled "
            "before that date see the future and rows labelled after it see "
            "stale data" % defn.pinned_knowledge_time,
        )
    return None


def _rule_lag_exceeds_window(defn, spec, declared):
    """Not leakage - skew. If facts typically arrive later than the window is
    wide, the window is usually empty at serving time but full at training
    time, because by the time anyone backfills the training set the late data
    has landed. Same divergence, arrived by a different road."""
    if (
        defn.window_days is not None
        and spec.typical_ingestion_lag_days >= defn.window_days
    ):
        return Finding(
            defn.name,
            "INGESTION_LAG_EXCEEDS_WINDOW",
            WARNING,
            "attribute %r arrives %.1f days late on average but the window is "
            "%.1f days wide; this feature will be empty in production and "
            "full in a backfilled training set"
            % (defn.attribute, spec.typical_ingestion_lag_days, defn.window_days),
        )
    return None


def _rule_unowned_attribute(defn, spec, declared):
    """The catalogue gap, surfaced instead of swallowed. An undeclared
    attribute cannot be checked for restatability, so the detector says so
    rather than implying the feature was verified."""
    if not declared:
        return Finding(
            defn.name,
            "ATTRIBUTE_NOT_IN_CATALOGUE",
            WARNING,
            "attribute %r has no catalogue entry, so restatement behaviour is "
            "unknown and only the generic time-predicate rules were applied"
            % defn.attribute,
        )
    return None


_RULES = (
    _rule_knowledge_time_unbounded,
    _rule_restated_attribute_unbounded,
    _rule_future_event_window,
    _rule_pinned_snapshot,
    _rule_lag_exceeds_window,
    _rule_unowned_attribute,
)


def format_report(findings_by_feature: dict) -> str:
    lines = []
    for name in sorted(findings_by_feature):
        findings = findings_by_feature[name]
        verdict = (
            "BLOCKED"
            if any(f.severity == CRITICAL for f in findings)
            else ("WARN" if findings else "OK")
        )
        lines.append("%-28s %s" % (name, verdict))
        for finding in findings:
            lines.append("    %s" % finding)
    return "\n".join(lines)
