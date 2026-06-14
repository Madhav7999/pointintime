# pointintime

A feature store where point-in-time correctness is enforced, proved by a planted leaking feature the store refuses to serve.

Implementation note. The runnable core substitutes the spec's stack with standard-library equivalents: SQLite (bitemporal fact table, as-of reads via window functions) stands in for DuckDB; an in-process StreamingMaterialiser replaying the fact log under a watermark stands in for Kafka/Redpanda + Redis; a hand-written logistic regression and rank AUC stand in for sklearn; unittest replaces pytest. There is no FastAPI service or Docker setup. Code lives flat in src/ (store, registry, leakage, online, generator, experiment, model, demo) rather than the layout sketched below.

AI / ML Engineering - Fintech - Credit Decisioning

Primary language: Python
Tags: feature-store, point-in-time, data-leakage, streaming, kafka, sql

---

## The problem

A credit model scores 0.91 AUC in training and 0.68 in production. The cause is almost always the same: a feature was computed using data that did not exist at decision time - an account balance as of today joined to a decision made eight months ago. The model learned the future. This costs real money and is nearly invisible in a standard train/test split.

## The differentiator

Enforces point-in-time-correct joins as the only way to retrieve training data, and proves it with a leakage test that deliberately constructs a future-leaking feature and asserts the store refuses to serve it. It adds an automated train/serve skew check comparing offline and online values for the same entity and timestamp. A generic feature store provides a pleasant API for storing features and leaves point-in-time correctness as a convention the user must remember - which is to say, as a bug waiting for a deadline.

This is the core principle of the project. Everything else in this repo exists to make it true and to prove it.

## Data

A documented synthetic generator producing entity event streams - transactions, applications, repayments - with a planted leaking feature whose predictive power is entirely spurious, giving the leakage test a definite pass/fail. Structure informed by the openly available Lending Club historical data.

No paid API key is required to run or demo this project. Where a paid service would add value it is wired as an optional enhancement behind an interface with an offline mock as the default implementation.

## Stack

- Python
- DuckDB for the offline store; Redis for the online store
- Kafka / Redpanda for streaming materialisation
- SQL (as-of joins are the core artefact)
- FastAPI, Docker, CI, pytest

## Core capabilities

- Feature definitions declaring entity, event timestamp, aggregation window and freshness SLA
- Offline retrieval via as-of joins with a mandatory label-timestamp argument - there is no default
- Streaming materialisation to the online store with watermark handling for late events
- Train/serve skew detector comparing offline and online values on sampled entities
- Feature lineage and a registry carrying ownership and deprecation state

## Repository layout

```
src/registry/
src/offline/
src/online/
src/skew/
generator/
test/leakage/
```

## Build plan

1. Generator with a planted leaking feature. Fit a model on the naive join and enjoy the 0.91 - it is the point.
2. As-of join retrieval with a mandatory label timestamp. Making it mandatory is a design decision; defend it.
3. Streaming materialisation with watermarks, then the skew detector.
4. Freshness SLAs and the serving gate last.

## Testing strategy

Assert the planted leaking feature produces high AUC under a naive join and is rejected by the store. Assert offline/online value parity within tolerance across 10,000 sampled entity-timestamps. Assert that late-arriving events do not retroactively change already-served feature values - the subtle streaming bug that reintroduces leakage through the back door.

Tests assert correctness, not merely that the code runs. A green suite on this repo is a claim about behaviour under adversarial conditions; treat any test that would pass against a deliberately broken implementation as a bug in the test.

## Quality & safety layer

Freshness SLA violations block serving rather than silently returning stale features - a stale feature served as current is the online equivalent of leakage.

## Measurable outcome

Design target: offline AUC matches production AUC - the 0.91 was leakage, and the store makes that class of error structurally impossible. Measured on the synthetic data (see below): the naive join reports 0.912 offline, the same model scores 0.654 on production inputs, and the point-in-time model scores 0.701 offline, the number production can actually expect.

## Measured results

From python -m src.demo (seed 7, 400 synthetic applicants, 11,434 facts, 8,234 transactions of which 712 = 8.6% arrive late; chronological 60/40 split):

| Model / path | Test AUC |
|---|---|
| Point-in-time features only | 0.701 |
| Naive latest-value join (offline, leaks restated income) | 0.912 |
| Same leaking model fed production (point-in-time) inputs | 0.654 |

- The leak inflates offline AUC by +0.259 over what ships. Over 5 seeds (250 applicants each, run in the test suite) the offline-minus-deployed gap is 0.225 mean, SE 0.024. Deployed AUC is statistically indistinguishable from the honest model (0.715 vs 0.709): all the leak's power is spurious.
- The registry refuses income_current_value (KNOWLEDGE_TIME_UNBOUNDED, RESTATED_ATTRIBUTE_WITHOUT_PIT_BOUND). The negative control income_at_application, which reads the same restatable column with a point-in-time bound, passes.
- Train/serve skew over 10,000 sampled entity-timestamps x 4 features: knowledge-time watermark stream 0 / 40,000 mismatches (exact equality); event-time watermark stream (the back-door bug) 3,492 / 40,000 (8.73%), concentrated in income_at_application (1,570) and the 90-day transaction features (902 each).
- Mutation check: forcing the store's as-of cutoff to "latest" (a naive latest-value join) fails 4/18 store tests, 4/12 streaming tests and 8/11 leak-measurement tests.
- Suite: 59 tests, about 35-45 s; demo about 8 s.

### Limitations

- Synthetic data only; the planted restatement gap (11,000 USD, SD 3,000) is deliberately large, so the 0.91 inflation is a property of the generator, not an estimate for real lending data.
- The detector is definitional: it is only as good as the attribute catalogue. An attribute not marked restatable gets only the generic rules.
- SQLite + in-process replay, not DuckDB/Redis/Kafka: no concurrency, partitions, or real out-of-order network delivery; parity is exact because both paths share the same aggregation code.
- Future-dated and knowledge-before-event rows are rejected or excluded at read/append time; the generator does not plant them in bulk (tests inject them directly).
- No FastAPI service, lineage UI, or deprecation workflow beyond owner fields.

## Interview questions this project answers

- What is a point-in-time correct join and why does it matter?
- How do late-arriving events cause leakage?
- What is train/serve skew and how would you detect it?

## What this deliberately is not

- Not a Feast clone. It implements the one guarantee most feature stores leave to convention.
- Not a model-training framework - it feeds one.

## Run it now

```bash
python -m unittest discover -s tests -v   # the suite
python -m src.demo                        # the 60-second artefact
```

Requires Python 3.11+. The runnable core uses only the standard library (including sqlite3), so there is nothing to install.

## Getting started

```bash
git clone https://github.com/madhavmeesala/pointintime.git
cd pointintime
python -m unittest discover -s tests -v   # the suite
python -m src.demo                        # the 60-second artefact
```

Nothing to install - the runnable reference has no dependencies.

<details>
<summary>Target workflow for the full build (not yet implemented)</summary>

These commands describe the production stack this project grows into. None of them work in this repository today.

```bash
# cd pointintime
# docker compose up -d          # Redpanda + Redis
# pip install -e .
# python -m generator --entities 200000
# pytest test/leakage           # naive join leaks; the store refuses
# python -m src.skew check
```

</details>

Docker is supported but optional - every path above works on a plain Windows/macOS/Linux laptop without a cloud account.

## Definition of done

- The differentiator above is implemented, and a test proves it
- The measurable outcome is produced by a command anyone can run
- README explains the one decision a generic version gets wrong
- CI runs the full suite on every push and is green on main
- A recruiter can see the headline artefact in under 60 seconds

## Maintainer

Madhav Meesala is a Software Engineer with over 4 years of experience in building scalable distributed systems and financial services. This project reflects his focus on MLOps, data engineering, and ensuring the integrity of production machine learning pipelines.

Email: madhavmeesala@gmail.com
GitHub: github.com/madhavmeesala

## Licence

MIT - see [LICENSE](LICENSE).