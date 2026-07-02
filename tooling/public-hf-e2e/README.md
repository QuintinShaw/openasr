# Public-HF E2E

This smoke verifies the public distribution path without using Hugging Face
credentials:

1. build or use an `openasr` CLI binary;
2. fetch the public catalog;
3. `openasr pull` a real public `.oasr` pack into an isolated `OPENASR_HOME`;
4. transcribe a committed audio fixture with `--backend native --model-pack`;
5. fail if the transcript is empty, non-textual, or appears to come from the
   mock backend.

Run:

```bash
tooling/public-hf-e2e/run.sh
```

Useful overrides:

```bash
OPENASR_PUBLIC_HF_E2E_MODEL=moonshine-tiny:q8 \
OPENASR_PUBLIC_HF_E2E_AUDIO=fixtures/jfk.wav \
tooling/public-hf-e2e/run.sh --keep
```

For a copyable validation record, write a redacted evidence block:

```bash
OPENASR_PUBLIC_HF_E2E_BIN=target/debug/openasr \
tooling/public-hf-e2e/run.sh \
  --summary-md tmp/public-hf-e2e/validation.md
```

Use `--strict-evidence` for the final public-HF release gate:

```bash
OPENASR_PUBLIC_HF_E2E_BIN=target/debug/openasr \
tooling/public-hf-e2e/run.sh \
  --strict-evidence \
  --summary-json tmp/public-hf-e2e/summary.json \
  --summary-md tmp/public-hf-e2e/validation.md
```

Strict evidence rejects dry-runs, local or non-canonical catalog URLs, and runs
without `--summary-json`. `--summary-md` is optional and useful for
a local validation evidence log, but JSON is required so the summary can be
machine-checked.
The summary records only public or basename-level data: model, canonical catalog
status, audio fixture filename, installed pack filename/hash/size, transcript
length, transcript letter count, and a transcript preview. The gate audit uses
all three transcript fields and rejects empty previews or mock-backend
placeholders. It does not record `OPENASR_HOME`, absolute pack paths, token
environment variables, or local directory names.

The manual/scheduled GitHub workflow writes both summary formats to
`tmp/public-hf-e2e/` and uploads them as the `public-hf-e2e-evidence` artifact.
For a final release gate, run the workflow manually with `strict_evidence=true`.

The default model is intentionally the smallest public pack. The workflow is
kept outside push/PR CI because it performs real network downloads and native
model execution.

No-network helper tests:

```bash
python3 -m unittest discover -s tooling/public-hf-e2e -p 'test_*.py'
```
