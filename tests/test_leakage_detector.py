"""The detector reasons about DEFINITIONS, not about accuracy.

Two things are being proved here, and the second is the harder one:

  1. the planted leak is flagged and cannot be registered;
  2. the NEGATIVE CONTROL - a legitimate feature reading the very same
     restatable attribute, which a column-level blocklist would flag - is
     not flagged, and no verdict changes when the features are renamed.
"""

import inspect
import unittest

from src import generator
from src.leakage import CRITICAL, WARNING, audit_definition, is_leaking
from src.registry import (
    AS_OF_LABEL,
    LATEST,
    PINNED_SNAPSHOT,
    AttributeSpec,
    FeatureDefinition,
    FeatureRegistry,
    LeakageError,
)
from src.store import PointInTimeStore
from src.timeline import ts


def catalogue():
    return {
        "annual_income": AttributeSpec(
            "annual_income", restatable=True, typical_ingestion_lag_days=0.5,
            owner="underwriting",
        ),
        "credit_utilisation": AttributeSpec(
            "credit_utilisation", restatable=False, typical_ingestion_lag_days=1.0,
            owner="bureau",
        ),
        "disputed_charge": AttributeSpec(
            "disputed_charge", restatable=False, typical_ingestion_lag_days=21.0,
            owner="payments",
        ),
    }


def rules(findings):
    return [f.rule for f in findings]


def critical(findings):
    return [f.rule for f in findings if f.severity == CRITICAL]


class PlantedLeakTests(unittest.TestCase):
    def test_the_planted_leak_is_flagged_critical(self):
        findings = audit_definition(generator.leaking_definition(), catalogue())
        self.assertIn("KNOWLEDGE_TIME_UNBOUNDED", critical(findings))
        self.assertIn("RESTATED_ATTRIBUTE_WITHOUT_PIT_BOUND", critical(findings))

    def test_the_store_refuses_to_register_it(self):
        registry = FeatureRegistry(PointInTimeStore())
        for spec in catalogue().values():
            registry.declare(spec)
        with self.assertRaises(LeakageError) as caught:
            registry.register(generator.leaking_definition())
        self.assertIn("KNOWLEDGE_TIME_UNBOUNDED", str(caught.exception))
        self.assertNotIn(generator.LEAKING_FEATURE, registry.features)
        self.assertIn(generator.LEAKING_FEATURE, registry.quarantine)

    def test_a_quarantined_feature_cannot_be_computed_by_name(self):
        dataset = generator.generate(seed=3, n_entities=5)
        entity, label_time, _ = dataset.rows[0]
        with self.assertRaises(LeakageError):
            dataset.registry.compute(generator.LEAKING_FEATURE, entity, label_time)

    def test_the_only_difference_from_the_control_is_the_knowledge_policy(self):
        # If these two definitions differed in any other field, the detector
        # could be passing this suite by accident.
        leak = generator.leaking_definition()
        control = next(
            d for d in generator._honest_definitions()
            if d.name == "income_at_application"
        )
        differing = [
            field
            for field in ("attribute", "aggregation", "window_days",
                          "event_time_offset_days")
            if getattr(leak, field) != getattr(control, field)
        ]
        self.assertEqual(differing, [])
        self.assertNotEqual(leak.knowledge_policy, control.knowledge_policy)


class NegativeControlTests(unittest.TestCase):
    """The feature that looks guilty and is innocent."""

    def setUp(self):
        self.control = next(
            d for d in generator._honest_definitions()
            if d.name == "income_at_application"
        )

    def test_the_negative_control_is_not_flagged_at_all(self):
        findings = audit_definition(self.control, catalogue())
        self.assertEqual(
            findings,
            [],
            "income_at_application reads the same restatable attribute as the "
            "leak; flagging it would make the gate noise",
        )
        self.assertFalse(is_leaking(self.control, catalogue()))

    def test_the_negative_control_registers_cleanly(self):
        registry = FeatureRegistry(PointInTimeStore())
        for spec in catalogue().values():
            registry.declare(spec)
        registry.register(self.control)
        self.assertIn("income_at_application", registry.features)

    def test_a_strong_but_legitimate_feature_is_not_flagged(self):
        # Prior utilisation is a huge predictor of default. An accuracy-based
        # detector flags it; a definitional one has no reason to.
        strong = FeatureDefinition(
            name="utilisation_max_365d",
            attribute="credit_utilisation",
            aggregation="max",
            window_days=365.0,
        )
        self.assertEqual(audit_definition(strong, catalogue()), [])


class VerdictDependsOnFieldsNotNamesTests(unittest.TestCase):
    def test_renaming_the_leak_does_not_launder_it(self):
        disguised = generator.leaking_definition().replace(
            name="boring_dimension_lookup", description="nothing to see here"
        )
        self.assertTrue(is_leaking(disguised, catalogue()))

    def test_an_alarming_name_does_not_condemn_a_correct_definition(self):
        alarming = FeatureDefinition(
            name="income_leak_v2_final_DO_NOT_SHIP",
            attribute="annual_income",
            aggregation="last",
            knowledge_policy=AS_OF_LABEL,
            tags=("future", "leak", "hindsight"),
        )
        self.assertEqual(audit_definition(alarming, catalogue()), [])

    def test_fixing_the_policy_clears_the_finding(self):
        fixed = generator.leaking_definition().replace(knowledge_policy=AS_OF_LABEL)
        self.assertEqual(audit_definition(fixed, catalogue()), [])

    def test_the_detector_has_no_access_to_data_or_labels(self):
        # A signature check, because this is the claim the README makes: the
        # verdict cannot be a function of measured accuracy if the function
        # cannot see a model, a label or a store.
        params = list(inspect.signature(audit_definition).parameters)
        self.assertEqual(params, ["defn", "attributes"])


class OtherLeakMechanismTests(unittest.TestCase):
    def test_future_event_window_is_caught_even_with_a_correct_policy(self):
        defn = generator.future_window_definition()
        self.assertEqual(defn.knowledge_policy, AS_OF_LABEL)
        findings = audit_definition(defn, catalogue())
        self.assertEqual(critical(findings), ["FUTURE_EVENT_WINDOW"])

    def test_pinned_snapshot_is_caught(self):
        defn = FeatureDefinition(
            name="income_snapshot_2024",
            attribute="credit_utilisation",
            knowledge_policy=PINNED_SNAPSHOT,
            pinned_knowledge_time=ts(400),
        )
        self.assertIn("PINNED_KNOWLEDGE_SNAPSHOT", critical(audit_definition(defn, catalogue())))

    def test_pinned_snapshot_requires_a_time(self):
        with self.assertRaises(ValueError):
            FeatureDefinition(
                name="x", attribute="credit_utilisation",
                knowledge_policy=PINNED_SNAPSHOT,
            )

    def test_unknown_policy_is_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            FeatureDefinition(name="x", attribute="y", knowledge_policy="whenever")


class SkewWarningTests(unittest.TestCase):
    """Warnings are advisory: they describe train/serve skew, not leakage,
    and must not block registration - a gate that blocks on warnings is a
    gate people route around."""

    def test_lag_exceeding_window_warns_but_does_not_block(self):
        defn = FeatureDefinition(
            name="disputes_7d",
            attribute="disputed_charge",
            aggregation="count",
            window_days=7.0,
        )
        findings = audit_definition(defn, catalogue())
        self.assertEqual(rules(findings), ["INGESTION_LAG_EXCEEDS_WINDOW"])
        self.assertEqual(findings[0].severity, WARNING)
        registry = FeatureRegistry(PointInTimeStore())
        for spec in catalogue().values():
            registry.declare(spec)
        registry.register(defn)
        self.assertIn("disputes_7d", registry.features)

    def test_an_uncatalogued_attribute_is_reported_as_unverified(self):
        defn = FeatureDefinition(name="mystery", attribute="ghost_column")
        findings = audit_definition(defn, catalogue())
        self.assertEqual(rules(findings), ["ATTRIBUTE_NOT_IN_CATALOGUE"])
        self.assertEqual(critical(findings), [])

    def test_a_catalogued_attribute_is_not_reported_as_unverified(self):
        defn = FeatureDefinition(name="util", attribute="credit_utilisation")
        self.assertEqual(audit_definition(defn, catalogue()), [])


if __name__ == "__main__":
    unittest.main()
