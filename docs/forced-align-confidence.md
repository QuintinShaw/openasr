# Forced-aligner acoustic confidence

Qwen3-ForcedAligner-0.6B is a non-autoregressive **timestamp classifier**,
not a CTC acoustic model. For each manuscript word it emits two classify-head
rows (start and end). Each row is a softmax over the 80 ms timestamp grid
(`classify_num` bins). The runtime still picks the timestamp with
`stable_timestamp_bin`; this document only scores that already-chosen bin.

## Criterion

**Score** = mean, over every start/end boundary, of the chosen-bin
log-softmax.

This is the NAR analog of a CTC forced-path average log-probability
(Graves, Fernández, Gomez, Schmidhuber, ICML 2006, *Connectionist Temporal
Classification*) and of max-softmax-probability OOD detection (Hendrycks &
Gimpel, ICLR 2017, *A Baseline for Detecting Misclassified and
Out-of-Distribution Examples*). The classify head cannot emit a CTC blank
or a free-decode token posterior; those signals do not exist on this pack.

**Alternatives considered**

| Statistic | Why not shipped |
| --- | --- |
| Per-word posterior median | A 50/50 partial match can keep the median on the matching half. |
| Lower-quartile (p25) of per-word means | Larger partial-match gap on this fixture set, but it is an order statistic rather than the path-average log-prob used in the CTC / MFA literature. Mean already separates every pair below. |
| Forced-path vs free-decode gap | Would require a second unconstrained decode of a different model family. Out of scope. |
| WER / string overlap | Explicitly forbidden by #391. |

The geometric gates (collapsed bins, >50% zero-duration words,
non-monotonic timestamps) are unchanged and still run first.

## Threshold

```text
MIN_MEAN_CHOSEN_BIN_LOG_PROB = -1.00
```

Calibrated 2026-09-07 on this host (Apple M1, CPU graph, isolated
`OPENASR_HOME`, shipped `qwen3-forced-aligner-0.6b:q4_k` object
`sha256:5b36662d373cbee279f168c2f88700a59a93246584bf3a5ade9e800e41c7807b`).

| Manuscript | Audio | Mean log-prob | p25 word | Min word | Words | Decision at −1.00 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Correct JFK English | `fixtures/jfk.wav` | −0.360 | −0.481 | −0.812 | 22 | admit |
| Correct zh_sample Chinese | `fixtures/zh_sample.wav` | −0.051 | −0.067 | −0.332 | 74 | admit |
| Mixed EN/ZH golden | `fixtures/en_zh_mixed.wav` | −0.118 | −0.205 | −0.431 | 40 | admit |
| Unrelated English recipe | `fixtures/jfk.wav` | −2.304 | −2.828 | −3.335 | 18 | reject |
| zh_sample Chinese manuscript | `fixtures/jfk.wav` | −2.744 | −3.361 | −3.985 | 74 | reject |
| JFK English | `fixtures/zh_sample.wav` | −2.464 | −2.905 | −3.295 | 22 | reject |
| Unrelated English recipe | `fixtures/zh_sample.wav` | −2.439 | −2.977 | −3.402 | 18 | reject |
| JFK first half + recipe tail | `fixtures/jfk.wav` | −1.498 | −2.243 | −3.464 | 32 | reject |

**Separation margin**

- Worst matching mean (−0.360) is **0.64 nats above** the threshold.
- Closest mismatch is the partial manuscript (−1.498), **0.50 nats below**
  the threshold.
- Full-mismatch ceiling (−2.304) is **1.30 nats below** the threshold.

A score equal to the threshold is admitted (same convention as the 50%
zero-duration geometric gate). Failure returns
`WordTimestampAlignmentFailed` (HTTP 400 / CLI runtime failure) with the
score and threshold in the message, the same error class as the geometric
gates.

## Known limits

- Near-miss manuscripts (a few substituted or extra words on an otherwise
  matching script) were not in the calibration set. They may score between
  the matching cluster and −1.00.
- The score measures how peaked the timestamp classifier is, not token
  identity. A manuscript that happens to place confident (but wrong)
  boundaries could in principle pass; that was not observed on the
  fixture cross-pairs above.
- Japanese / Korean remain fail-closed by the existing morphology guard
  before this score is computed.
