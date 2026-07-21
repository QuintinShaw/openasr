# Model release audits

One completed audit form per model family, named `<family>.md` where
`<family>` is the `family` value from
`tooling/publish-model/models-core.toml`. Start every form by copying
[TEMPLATE.md](TEMPLATE.md).

## Why

The form exists so every model release ships in its best known state --
maximum performance, complete backend coverage, verified correctness -- and so
every consciously skipped optimization is on the record with a justification
and an unlock condition instead of silently forgotten. It is a means to peak
release quality, not paperwork: filling one should take an afternoon.

## The rules

- **When to fill:** before a new model family enters the release flow, i.e.
  before its first catalog entry flips `public:true`. Effective 2026-07, this
  is mandatory for every new family.
- **Enforcement:** the publish pipeline fails closed.
  `tooling/publish-model/scripts/_manifest.py --public` (also reached via
  `regenerate_all.sh --public`) refuses to write a public catalog entry when
  `docs/model-audits/<family>.md` is missing, still contains `<!-- TODO:fill -->`
  markers, or lost any of the ten numbered sections. See
  `tooling/publish-model/scripts/audit_form.py`.
- **Ships with the release:** the completed form is part of the release; keep
  it updated when a dimension's status changes (e.g. a Deferred item lands).
- **Existing families:** families public before this policy are grandfathered
  in `audit_form.py` (`PRE_AUDIT_FAMILIES`) and get backfilled on the rolling
  audit matrix. When a family's form lands, remove it from that set -- the set
  only shrinks, never grows.

## Filling guidance

Each item is `Supported` / `Not applicable` / `Deferred`. Should-support items
MUST be `Supported`; `Not applicable` and `Deferred` both require a detailed
justification, and `Deferred` additionally requires the unlock condition that
flips it. Cite evidence (test names, bench runs, gate scripts) rather than
asserting -- the form is only as useful as its audit trail.
