# Model release audit: <family> <!-- TODO:fill -->

> **Policy.** Should-support items MUST be `Supported`; anything else requires a
> detailed justification and an explicit unlock condition. This form ships with
> the model release. A family without a completed form does not enter the
> release flow: `tooling/publish-model/scripts/_manifest.py --public` fails
> closed on a missing or half-filled form.

| Field | Value |
| --- | --- |
| Family (`models-core.toml` `family`) | <!-- TODO:fill --> |
| Models covered | <!-- TODO:fill --> |
| Auditor / date | <!-- TODO:fill --> |
| Core version + commit audited | <!-- TODO:fill --> |
| Bench hardware | <!-- TODO:fill --> |

**How to fill.** Status is exactly one of:

- `Supported` -- implemented and verified for this family in this repo. Cite
  the evidence (test name, bench run, code path).
- `Not applicable` -- architecturally impossible or meaningless for this
  family. Say why, so nobody re-derives it.
- `Deferred` -- applicable but intentionally not done yet. Give the detailed
  justification AND the unlock condition (what measurement, upstream change,
  or milestone flips it to Supported).

Replace every `<!-- TODO:fill -->` marker; the release gate rejects any
leftover marker. Do not delete or rename the ten numbered section headings; the
gate checks all ten. Keep entries terse -- one form should take an afternoon,
not a week. The goal is that every release ships in its best known state, with
every consciously skipped optimization on the record.

## 1. Graph & scheduling

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| Graph reuse / persistent session (no per-request graph rebuild) | <!-- TODO:fill --> | |
| Op fusion opportunities reviewed (norm+matmul, QKV, rope, ...) | <!-- TODO:fill --> | |
| Batching / serve-batch path | <!-- TODO:fill --> | |
| Encode-decode pipelining | <!-- TODO:fill --> | |
| Arena / gallocr reuse across steps (no per-step allocator churn) | <!-- TODO:fill --> | |

## 2. Precision & quantization

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| KV cache quantization | <!-- TODO:fill --> | |
| Activation precision policy chosen deliberately (f32 vs f16) | <!-- TODO:fill --> | |
| Keep-quantized matmul (native Q blocks bound, no load-time dequant; RAM orders q4 < q8 < fp16) | <!-- TODO:fill --> | |
| Quant tiers complete (q4_k / q8_0 / fp16) | <!-- TODO:fill --> | |

## 3. Memory & data movement

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| mmap weight loading | <!-- TODO:fill --> | |
| Resident pool reuse across requests (weights stay resident) | <!-- TODO:fill --> | |
| View contiguity tradeoffs audited (`cont`/copy nodes justified) | <!-- TODO:fill --> | |

## 4. Decode algorithms

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| Greedy logits shortcuts (argmax path skips needless softmax/sort work) | <!-- TODO:fill --> | |
| Speculative decode: per-family verdict recorded (do it, defer it, or dead) | <!-- TODO:fill --> | |
| CTC blank-skip fast path (CTC families; otherwise Not applicable) | <!-- TODO:fill --> | |
| Decode guards are zero-cost on the hot path (degenerate-loop guard etc.) | <!-- TODO:fill --> | |

## 5. Frontend & IO

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| Mel/fbank frontend SIMD + parallelized | <!-- TODO:fill --> | |
| Zero-copy audio path (no avoidable resample/copy hops) | <!-- TODO:fill --> | |
| VAD cost measured and accounted | <!-- TODO:fill --> | |

## 6. Platform-specific

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| Metal command batching + wired memory budget respected | <!-- TODO:fill --> | |
| CPU thread pool sized for P/E cores | <!-- TODO:fill --> | |
| Accelerate/BLAS used where it wins | <!-- TODO:fill --> | |

## 7. Backend coverage matrix

Every cell must be answered. An unsupported backend is acceptable ONLY with a
justification and an unlock plan -- "nobody tried" is not a justification.
Golden-verified means byte/parity fixtures pass ON that backend;
utilization-measured means the GPU weight placement gate (or an equivalent
profile) proved the compute actually runs there (golden output alone cannot,
see `docs/design/gpu-weight-placement.md`).

| Backend | Supported? | Golden-verified? | Utilization measured? | Justification + unlock plan if unsupported |
| --- | --- | --- | --- | --- |
| CPU | <!-- TODO:fill --> | <!-- TODO:fill --> | <!-- TODO:fill --> | |
| Metal | <!-- TODO:fill --> | <!-- TODO:fill --> | <!-- TODO:fill --> | |
| CUDA | <!-- TODO:fill --> | <!-- TODO:fill --> | <!-- TODO:fill --> | |
| Vulkan | <!-- TODO:fill --> | <!-- TODO:fill --> | <!-- TODO:fill --> | |
| HIP | <!-- TODO:fill --> | <!-- TODO:fill --> | <!-- TODO:fill --> | |

## 8. Correctness & quality

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| WER vs fp16 measured for every shipped quant tier | <!-- TODO:fill --> | |
| Golden coverage includes long audio AND a cross-backend parity fixture | <!-- TODO:fill --> | |
| Official decode parameters honored (suppression, stop tokens, upstream reference settings) | <!-- TODO:fill --> | |
| Long-audio degradation checked (repetition, drift, truncation) | <!-- TODO:fill --> | |

## 9. Resource limits & fail-closed

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| Max audio length / context budget derived and over-limit behavior fails closed | <!-- TODO:fill --> | |
| Streaming first-token latency floor documented (chunk accumulation math; streaming families, otherwise Not applicable) | <!-- TODO:fill --> | |
| KV growth rate per audio second known | <!-- TODO:fill --> | |
| Metal wired-memory profile captured | <!-- TODO:fill --> | |
| Multi-session scaling behavior known (server concurrency) | <!-- TODO:fill --> | |
| Energy footprint noted (battery-relevant platforms) | <!-- TODO:fill --> | |

## 10. Engineering completeness

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| `warm_up` is a real implementation, not a stub | <!-- TODO:fill --> | |
| Reference dumper exists for this family | <!-- TODO:fill --> | |
| Registry / catalog / docs wired (MODEL_ONBOARDING checklist done) | <!-- TODO:fill --> | |
| Peer benchmark recorded (table below, all fields) | <!-- TODO:fill --> | |

### Peer benchmark record

Record enough that anyone can re-run this comparison later. "Faster than X" is
not auditable without the exact peer version, model build, audio, and machine.

| Field | Value |
| --- | --- |
| Peer project (name + commit or version) | <!-- TODO:fill --> |
| Peer model + quant build | <!-- TODO:fill --> |
| Peer program version | <!-- TODO:fill --> |
| Test audio (file, duration, language) | <!-- TODO:fill --> |
| Machine (chip, RAM, OS) | <!-- TODO:fill --> |
| Peer numbers (RTF / peak memory / utilization) | <!-- TODO:fill --> |
| OpenASR numbers (RTF / peak memory / utilization) | <!-- TODO:fill --> |

## Known dead ends (do not re-litigate)

Verdicts that apply to this family, so future work does not re-run dead
investigations. Repo-wide precedents to inherit where relevant: F16 activation
on Apple M1 (encoder-only gave zero win, cast economics lock the trunk;
verdict 2026-07-14); qwen speculative decode (acceptance alpha ~= 0.05, judged
dead). Add family-specific verdicts with the measurement behind each; write
"None yet" if the family has none.

| Dead end | Verdict / evidence | Date |
| --- | --- | --- |
| <!-- TODO:fill --> | | |
