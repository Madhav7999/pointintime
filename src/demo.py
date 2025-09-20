"""60-second artefact: the leak, the refusal, the skew check. ASCII only."""

from __future__ import annotations

import time

from . import generator
from .experiment import run_experiment
from .leakage import audit_definition, format_report
from .online import (EVENT_GATE, KNOWLEDGE_GATE, sample_entity_timestamps,
                     streaming_parity)
from .registry import LeakageError
from .timeline import ts


def main() -> None:
    start = time.time()
    print("pointintime demo (seed 7, 400 applicants)")
    print("=" * 60)

    r = run_experiment(seed=7, n_entities=400)
    d = r["dataset"]
    print("\n[1] Planted data")
    print("  applicants=%d default_rate=%.3f store_rows=%d"
          % (r["n"], r["default_rate"], d.store.row_count()))
    print("  transactions=%d late_arriving=%d (%.1f%%)"
          % (d.transaction_count, d.late_transaction_count,
             100.0 * d.late_transaction_count / d.transaction_count))
    print("  income restated after decision: gap %.0f USD defaulters vs repaid"
          % d.truth["restatement_gap"])

    print("\n[2] Offline vs production AUC (chronological 60/40 split)")
    print("  honest PIT features only       AUC %.3f" % r["honest_auc"])
    print("  naive latest-value join        AUC %.3f  <- offline, leaks" % r["offline_auc"])
    print("  same model, production inputs  AUC %.3f  <- what ships" % r["deployed_auc"])
    print("  inflation from leak            %+.3f" % r["offline_minus_deployed"])

    print("\n[3] Store refuses the leaking feature")
    findings = {generator.LEAKING_FEATURE: audit_definition(
        generator.leaking_definition(), d.registry.attributes)}
    for name in generator.HONEST_FEATURES:
        findings[name] = audit_definition(d.registry.features[name], d.registry.attributes)
    print("  " + format_report(findings).replace("\n", "\n  "))
    try:
        d.registry.training_matrix([generator.LEAKING_FEATURE], d.rows[:1])
    except LeakageError as exc:
        print("  training_matrix -> LeakageError: %s" % exc)

    print("\n[4] Train/serve skew: 10000 sampled entity-timestamps x 4 features")
    names = list(generator.HONEST_FEATURES)
    samples = sample_entity_timestamps(d.store, 10000, ts(30), ts(530), seed=1)
    for gate, label in ((KNOWLEDGE_GATE, "knowledge-time watermark (correct)"),
                        (EVENT_GATE, "event-time watermark (the bug)")):
        fresh = generator.generate(seed=7, n_entities=400)
        rep = streaming_parity(fresh.registry, samples, names, gate)
        print("  %-36s mismatches %5d / %d (%.2f%%)"
              % (label, rep.mismatched, rep.checked, 100.0 * rep.mismatch_rate))
        for n in names:
            print("      %-24s %d" % (n, rep.per_feature[n]["skewed"]))

    print("\nelapsed %.1fs" % (time.time() - start))


if __name__ == "__main__":
    main()
