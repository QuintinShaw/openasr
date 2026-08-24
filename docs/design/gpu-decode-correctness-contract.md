# GPU decode correctness contract

Status: fail-closed software contract implemented; release remains blocked on
missing or stale real-hardware and product evidence.

Historical evidence baseline (the source snapshot on which the defect was
first proven, not a claim about the current release candidate):

- OpenASR source: `562f3fa7e498b3cd0e94908b99477351a0aa6ef1`
- vendored openasr-ggml pin: `0db3a287511085fb1564fea3f86839f9e40ca39e`

## Implementation checkpoint

The shared software migration is in place:

- reverse-gather ordinary `ARGMAX` is no longer a production authorization
  path; unproved GPU FullDevice lanes use `FullLogits` and `FreshGraph`;
- native `ARGMAX_FIRST` is capability-gated, and XASR/MiMo/SenseVoice retain
  their declared family-oracle semantics rather than inheriting a global tie
  rule;
- exact selected-device capability evidence, output plan, reuse mode, pack
  content identity, actual provider/device, scheduler state, and evidence
  revision are bound into request/activation receipts;
- same-graph dual output, fresh/reuse four-quadrant, backend-op, Layer-3, and
  release-matrix gates are separate; missing, stale, deferred, or mismatched
  cells fail closed;
- FireRed's test-only encoder twin now taps every bounded subsample seam before
  relative-position attention/depthwise/readback, so a complete run can name
  the first input, convolution, bias, ReLU, layout, or projection divergence;
  incomplete taps remain `insufficient_evidence`; and
- the Desktop companion contains a shipped plugin-switch runner and keeps a
  failed activation on the previous selected/LKG backend.

The evidence boundary remains strict. The committed CPU/Metal Layer-3 receipts
at the earlier checkpoint do not authorize a later release commit. This Apple
host has no CUDA, physical Vulkan, or HIP/ROCm lane, and its plugin host is
`legacy_static`, not the shipped Windows `neutral_dynamic` host. Those cells,
including HIP capture-on and the original Windows CUDA fp16 reproduction, are
not passes and remain non-activatable until release-bound receipts are returned.

This design complements [GPU weight placement](gpu-weight-placement.md),
[Decoder state and native memory planning](decoder-state-memory-planning.md),
[Runtime source preflight](runtime-source-preflight.md), and the
[Model-family lifecycle](model-family-lifecycle.md). Those documents remain
authoritative for placement, state sizing, immutable runtime sources, and
family inventory. This document owns a narrower question: when a decode path
selects a discrete token or code on a device, what proves that the result is the
same one the family's host oracle would have selected from complete logits or
scores?

## Executive decision

The observed FireRed AED failure is not evidence that inference silently moved
to the Intel integrated GPU. The diagnostic process loaded the CUDA plugin,
enumerated an RTX 4070 Laptop GPU, selected `CUDA0`, and then ran the FireRed
request in that process. The decoder emitted the same token eight times by
greedy step 7, the shared degenerate-loop guard truncated that decode, and the
request was later canceled. Dolphin completed in the same CUDA environment.

Dolphin's success is useful but narrow evidence. Dolphin does not use the
FireRed AED autoregressive device-top1 and reusable-KV path. It proves neither
that the FireRed logits were correct nor that the token selected from those
logits was correct.

The investigation found a separate, source-proven shared defect. OpenASR's
`top1_argmax_first_max_reversed` helper assumes that ordinary ggml `ARGMAX`
selects the last column when multiple values share the maximum. That assumption
holds for the current serial CPU and Metal implementations, but CUDA, Vulkan,
and the HIP-derived path use parallel reductions with no portable stable-last
contract. Reversing the vocabulary row and applying ordinary `ARGMAX` therefore
cannot implement OpenASR's deterministic first-maximum contract on those
backends.

This defect blocks capability activation even though exact maximum ties may be
uncommon; it also blocks a release if any already-selectable lane still uses the
defective path. It affects a shared production path used directly or indirectly
by multiple families. It must be fixed centrally. It is not, by itself, enough to attribute
the entire field failure: a constant token sequence can also result from stale
persistent output, reusable-KV state, mask or row-update defects, or incorrect
encoder/cross-attention values. The implementation must first run the
experiments in this document and preserve that epistemic boundary.

The target solution is:

1. keep complete logits/scores plus the family host oracle as the correctness
   baseline on every execution lane lacking proven compact-selection capability;
2. replace first-max reverse-gather emulation with native `ARGMAX_FIRST` on
   execution lanes that support and have validated it;
3. bring other device-side selectors, including XASR and MiMo RVQ, under the same
   declared host-oracle contract without changing their tie semantics;
4. separate device selection, persistent graph input/output refresh, and
   reusable-KV correctness into planner-internal runtime evidence;
5. validate backend operators, reusable graphs, and real model packs as three
   distinct layers; and
6. make the resulting evidence an exact-backend capability-activation gate,
   rather than post-release monitoring; public bytes may precede it only while
   they remain `PublishedInert` and unselectable.

No family-specific FireRed workaround satisfies this contract.

## Incident evidence and epistemic boundary

### Confirmed field facts

The supplied diagnostics establish the following sequence:

- the Windows host exposed an NVIDIA RTX 4070 Laptop GPU and an Intel integrated
  GPU;
- the installed CUDA plugin matched the `sm_89` device and loaded successfully;
- the daemon reported `best_backend=CUDA0`, selected `CUDA0`, and recorded
  `activated_provider=cuda`;
- the FireRed AED-L v2 fp16 request entered native GPU execution;
- the greedy loop detected eight repeats of one token at step 7 and kept only
  the first token;
- the request subsequently ended as canceled rather than as an OOM, unsupported
  operation, or typed placement failure; and
- Dolphin fp16 requests completed in the same CUDA-enabled process.

The degenerate-loop guard behaved correctly. It prevented a bad token stream
from expanding without bound. It did not create the repeated token, and changing
its threshold would hide evidence rather than fix the producing graph.

The request used long audio, but the diagnostic event does not prove that input
length caused the wrong token. Long-form execution slices requests before family
dispatch. Correctness investigation must use a representative 10-30 second
fixture first, hold the model and execution lane fixed, and compare per-step
values. Long-audio validation belongs after the short fixture has frozen the
implementation.

### What the diagnostics do not prove

The current logs do not contain enough information to determine whether the
first wrong value appeared in:

- the FireRed encoder;
- encoder readback or cross-K/V upload;
- the fresh decoder logits;
- the reusable decoder logits;
- the compact device-top1 selector; or
- a persistent i32 output that did not refresh between executions.

They also do not carry a per-request exact route receipt, a candidate fallback
journal, token IDs, logits hashes, cross-K/V hashes, or the cancellation origin.
Process-level backend selection is strong evidence that CUDA was active, but the
target runtime must eventually return a request-level execution receipt rather
than requiring this inference.

### Separate memory incident

A FireRed LLM q4 request in the same diagnostic set failed for a different
reason: constructing copied pack-weight buffers required about 5.09 GB of system
memory while the admitted remainder was about 3.92 GB. The memory broker
correctly rejected that allocation. It is covered by the runtime ownership and
atomic model-activation work and must not be used as evidence for or against the
FireRed AED token-selection defect.

## The confirmed first-max defect

### OpenASR's semantic contract

The shared greedy driver uses first-maximum selection: when several finite
logits share the largest value, it chooses the lowest vocabulary index. This is
a deterministic part of decode semantics, not an implementation preference.
Suppression, phrase bias, stop tokens, timestamps, and other consumers may also
require complete logits; a compact token hint is legal only when the selected
decode policy needs none of those values.

`crates/openasr-core/src/models/device_greedy_token.rs` encodes the first-max
index helpers and the current compact-token gate.
`crates/openasr-core/src/models/seq2seq_greedy_decode.rs` owns the shared greedy
selection and hint validation.

### The invalid cross-backend assumption

`GgmlCpuGraphBuilder::top1_argmax_first_max_reversed` in
`crates/openasr-core/src/ggml_runtime/cpu_graph.rs` implements first-max by:

```text
logits
  -> reverse vocabulary rows with GET_ROWS
  -> make the result contiguous
  -> ordinary ARGMAX
  -> map the reversed index back to the original vocabulary
```

This works only if ordinary `ARGMAX` chooses the last maximum in the reversed
row.

At the evidence baseline, backend behavior is:

| Backend | Ordinary `ARGMAX` tie behavior | Native `ARGMAX_FIRST` |
|---|---|---|
| CPU | stable last maximum | supported |
| Metal | stable last maximum | unsupported |
| CUDA | reduction-order dependent; no portable first/last guarantee | supported |
| Vulkan | reduction-tree dependent; no portable first/last guarantee | supported |
| HIP/ROCm | CUDA-derived reduction-order behavior | supported |

The distinction matters: strict `>` inside a serial loop preserves the first
maximum, but strict `>` inside a parallel reduction preserves whichever equal
candidate reaches a comparison first. Butterfly, tree, block-size, and lane
ordering are implementation details, not a semantic first-max contract.

Relevant implementations are:

- CPU ordinary argmax:
  `third_party/openasr-ggml/src/ggml-cpu/vec.h`
- CPU native first-max:
  `third_party/openasr-ggml/src/ggml-cpu/ops.cpp`
- CUDA/HIP argmax:
  `third_party/openasr-ggml/src/ggml-cuda/argmax.cu`
- Vulkan argmax shaders:
  `third_party/openasr-ggml/src/ggml-vulkan/vulkan-shaders/argmax.comp`
- Metal argmax:
  `third_party/openasr-ggml/src/ggml-metal/ggml-metal.metal`

The ggml backend-op suite already declares native `ARGMAX_FIRST` support for
CPU, CUDA, Vulkan, ROCm, and MUSA, and explicitly does not declare Metal
support: `third_party/openasr-ggml/tests/test-backend-ops.cpp`.

### Minimal counterexample

For one logits row:

```text
original = [2, 1, 5, 5]
```

OpenASR first-max must return token `2`.

The reverse helper produces:

```text
reversed = [5, 5, 1, 2]
```

Under the pinned CUDA reduction, lane 0 starts with a maximum and equal values
do not replace it. Ordinary `ARGMAX` can therefore return reversed index `0`.
Mapping that index back returns original token `3`, the last maximum, rather
than the required first maximum. Vulkan's parallel reduction has the same
absence of a portable last-max guarantee even though a particular row or workgroup
shape can happen to return the expected index.

The earlier-looking example `[1, 5, 5, 2]` is not a proof: for the pinned
parallel reduction and that shape, the reduction order can happen to map back
to the correct token. Correctness cannot be based on examples that depend on a
backend's current reduction tree.

This is a source-level contract failure: the helper requires stable last-max
ordinary argmax, while CUDA, Vulkan, and HIP provide no such contract. A
real-device test is still required to validate the compiled plugin and
persistent graph, but the semantic defect does not depend on reproducing the
field audio. The production comment in `cpu_graph.rs` that describes native
ggml argmax as universally returning the last exact maximum must be removed or
narrowed as part of the migration.

### Why exact ties remain in scope

A greedy decoder cannot assume that exact maximum ties never occur. They can be
created or made more likely by:

- quantized projections;
- zero or repeated rows;
- masked or suppressed logits;
- deterministic degenerate states;
- compact output tests and synthetic fixtures; and
- future kernels with different accumulation order.

Even if the field failure ultimately has a different first divergence, shipping
a known violation of deterministic decode semantics is unacceptable.

## Blast radius

### Direct shared device-top1 consumers

The current `device_greedy_step_output_mode` gate enables compact token output
only for an exact CUDA or Vulkan route, FullDevice placement, a direct GPU
runner, and scheduler-off execution. The following families consume that mode:

| Family | Compact selector | Current production reachability | Risk |
|---|---|---|---|
| FireRed AED | reversed gather + ordinary argmax | exact direct CUDA/Vulkan FullDevice | confirmed tie-contract defect; field failure also needs attribution |
| Cohere Transcribe | reversed gather + ordinary argmax | CUDA/Vulkan when no adapter, phrase bias, timestamps, or Cohere debug-token consumer requires logits | confirmed tie-contract defect |
| Moonshine | reversed gather + ordinary argmax | CUDA; Vulkan decoder defaults to CPU unless `OPENASR_MOONSHINE_ENABLE_DECODER_GPU` opts into the diagnostic GPU path | confirmed defect wherever the compact GPU path is reached |
| Granite Speech | reversed gather + ordinary argmax | exact direct CUDA/Vulkan FullDevice | confirmed tie-contract defect |
| SenseVoice | native `ARGMAX_FIRST` | capability-gated CUDA/Vulkan compact mode | not affected by reverse helper |

Cohere forces complete logits when execution bypasses serve-batch (including its
streaming path), unified runtime is disallowed, serve-batch is active, or an
adapter, phrase bias, word timestamps, or
`OPENASR_COHERE_DEBUG_TOKENS` consumer is present. Moonshine's streaming direct
path sets `skip_serve_batch` and forces complete logits; adapters, phrase bias,
and word timestamps do the same. Its offline serve-batch implementation is a
separate decode path and must be inventoried independently. Moonshine has no
Cohere-equivalent debug-token gate. These family/request restrictions are
semantic inputs to the shared output planner, not backend capability decisions.

### Qwen-shaped fused and reused top1

Qwen's whole-decoder fused logits head uses the same reverse helper. Its shared
runtime is reused by:

- Qwen3-ASR;
- MOSS Transcribe Diarize;
- FunASR Nano;
- MiMo ASR; and
- FireRed LLM.

The production direct call is in
`crates/openasr-core/src/models/qwen/llm_transformer.rs`; family wrappers call
its fused or reused top1 methods. The similarly named reverse helper in
`models/qwen/logits_head.rs` is test-only and is not an additional production
consumer.

Unlike the seq2seq `DeviceTop1` gate, Qwen-shaped fused/reused top1 is enabled by
`reusable_decode_graph_supported`, which currently means only
`is_gpu_class && !scheduler`. It is therefore already reachable on Metal and
HIP as well as CUDA and Vulkan. Metal's reverse + stable-last ordinary argmax
currently happens to implement first-max, but that backend-specific coincidence
must not survive as the long-term abstraction. CUDA and Vulkan violate the
helper's required last-max premise; HIP shares the CUDA implementation and is
the same risk. Qwen LoRA does not disable the fused selector, unlike the adapter
gates in Cohere and Moonshine.

### Other device-side discrete selectors

The contract covers every device-selected discrete decode value, not only the
reverse helper.

XASR's device head calls ordinary `top1_argmax`, while its host greedy oracle
uses Rust `max_by`, which currently selects the last equal maximum. XASR must
continue matching that family oracle; it must not be silently converted to the
seq2seq first-max rule. Ordinary parallel argmax cannot provide a portable
last-max contract on CUDA/Vulkan/HIP, so XASR uses `FullLogits` until a typed,
validated last-max device selection capability exists.

MiMo's RVQ graph also calls ordinary device argmax for codebook selection, while
its host `nearest_code` loop uses strict `>` and therefore keeps the first equal
code. RVQ codes are not transcription tokens, but a wrong code changes the
decode input and belongs to the same device-side discrete-selection correctness
surface. Its family oracle and three-layer evidence must be explicit.

The target planner therefore carries a semantic tie policy, at least
`FirstMaximum` or `LastMaximum`, supplied by the family algorithm. Native
`ARGMAX_FIRST` closes first-max consumers; it does not authorize changing a
last-max family oracle. A lane lacking a native operation proven for the
requested tie policy uses complete scores/logits plus the host oracle.

SenseVoice already uses native `ARGMAX_FIRST` for frame token IDs. It is not
subject to reverse-gather migration, but its cached graph can change between
full logits and frame token IDs while the current cache key contains only pack
and execution lane. The resolved output plan must enter that key if retained
graph topology differs.

### Reusable graphs without compact top1

Whisper uses reusable seq2seq graph infrastructure but currently returns complete
logits rather than consuming the shared `DeviceTop1` mode. It belongs in
persistent-graph output-refresh validation but not in the reverse-selector
migration.

Dolphin uses encoder, CTC prefix-beam, and attention rescoring. It does not use
the FireRed/Qwen autoregressive compact top1 path. Its successful CUDA result is
therefore not a parity result for the affected path.

### Four distinct risk classes

The implementation and review must not collapse these classes:

1. **Shared selector risk** - first-max semantics, compact output materialization,
   and stale i32 output.
2. **Shared reusable-graph risk** - refreshed token/row/position/mask inputs,
   in-place KV writes, topology reuse, and output refresh.
3. **Family graph risk** - FireRed relative-position attention, depthwise
   convolution, encoder readback, and cross-K/V construction.
4. **Provider kernel risk** - backend-specific operator, stride, capture,
   synchronization, and numerical behavior.

A shared selector fix does not prove a family encoder. A family transcript does
not prove the selector on other backends.

## Architectural defects that allowed the problem

### `is_gpu_class` is not a correctness capability

`reusable_decode_graph_supported` in
`crates/openasr-core/src/nn/decoder.rs` currently reduces the decision to:

```text
backend.is_gpu_class() && !uses_scheduler
```

The comments correctly record that CPU direct execution and the multi-backend
scheduler cannot safely reuse the in-place-KV graph. The remaining predicate is
still too broad. Metal, CUDA, Vulkan, and HIP have different operator support,
input refresh behavior, output allocation, synchronization, and graph-capture
semantics. "GPU class" is a placement category, not proof of persistent graph
correctness.

### Device selection and graph reuse are different capabilities

The current code mixes several independent statements:

- the backend can compute the decoder graph;
- the backend can select first-max on device;
- persistent graph inputs refresh correctly;
- persistent scalar outputs refresh correctly;
- in-place KV writes are visible to later steps;
- scheduler use is legal; and
- backend graph capture is legal.

One successful statement cannot authorize the others. The target contract must
represent them independently.

### Allocator markers are not graph-capture evidence

`set_input` and `set_output` are ggml allocator lifetime markers. They allow the
allocator to preserve input and output buffers across a graph allocation. They
do not prove that a captured executable graph observes changed host uploads,
changed row indices, changed masks, or changed output contents.

CUDA executable-graph capture is explicitly disabled by `build.rs` with
`GGML_CUDA_GRAPHS=OFF`. HIP capture is not a future possibility: OpenASR
currently defaults `OPENASR_HIP_GRAPHS` to true, the plugin build documents
`GGML_HIP_GRAPHS=ON`, and HIP shares the CUDA `USE_CUDA_GRAPH` implementation
path. CUDA capture-off evidence cannot authorize HIP capture-on behavior.

For HIP, `CaptureCompatibility` must describe the current capture-on lane. A
compact or reusable output plan remains unsupported until production-shape
persistent input/output and KV evidence passes with capture enabled. If the
implementation temporarily disables HIP capture to restore correctness, that is
an explicit execution-lane policy change with its own parity and performance
evidence, not an implicit fallback or a substitute for validating the shipping
capture-on configuration.

### A prior GPU-only repeated-token incident already exists

`tooling/qwen-gpu-parity/README.md` records a previous Qwen failure on Windows
HIP: CPU output was correct while the GPU produced garbled, repeated tokens due
to a provider-specific GQA kernel path. A synthetic single-op probe was rejected
as sufficient evidence because it could exercise the wrong shape or operation.
That incident and the FireRed report have different immediate candidates, but
they demonstrate the same architectural weakness: CPU correctness and generic
GPU placement do not prove real-model correctness on each provider.

## Target capability model

The target is a shared runtime capability contract. Family code declares model
semantics and consumes a resolved output plan. It does not parse provider names
or maintain a family-specific provider table.

### Independent capabilities

At minimum, the selected execution lane must expose typed statements equivalent
to:

- `NativeArgmaxFirstCapability`
  - native operator support;
  - supported input/output type and shape;
  - deterministic first-tie semantics;
  - validated output materialization and readback.
- `PersistentInputRefreshCapability`
  - token, row, position, mask, and other mutable inputs are observed on every
    graph execution.
- `PersistentOutputRefreshCapability`
  - scalar token and full-logits outputs reflect the current execution rather
    than retained storage.
- `ReusableKvGraphCapability`
  - in-place KV updates, causal masking, and subsequent reads are correct for the
    selected lane.
- `SchedulerCompatibility`
  - whether the graph can run under the multi-backend scheduler without losing
    refreshed inputs or violating placement.
- `CaptureCompatibility`
  - whether native executable-graph capture is disabled, unsupported, or proven
    for this graph contract.

These are planner-internal evidence dimensions and receipt fields, not six new
family-facing capability types. OpenASR already has the correct integration seam:
`GgmlBackendCapabilities` resolves backend facts once, and
`ResolvedFamilyRuntimeInput` carries the typed `GgmlNativeGqaCapability` into
family execution without requiring families to parse names such as `HIP0`.
Device selection must follow that pattern rather than create another registry or
expose a shallow capability-combination API to every family.

There is exactly one shared output-plan combiner:

```text
family tie/decode policy
+ request logits consumers
+ selected lane evidence
= FullLogits | NativeFirstMaxToken
```

A future family with a different host oracle, such as XASR last-max selection,
may resolve to another explicitly named native plan only after that tie policy
has a typed and proven lane capability. Family code sees the immutable resolved
plan and reuse mode; it does not independently combine the evidence dimensions.

`device_greedy_step_output_mode` currently matches
`ExecutionProvider::Cuda | Vulkan` directly. That provider allowlist must be
removed. Provider parsing and concrete `supports_op` probes belong at the shared
runtime boundary, consistent with the existing native-GQA design and the
repository's family/runtime boundary.

### Sources of capability

A capability is valid only when all of these agree:

1. the actual selected backend device reports support for the concrete ggml
   operation and shape;
2. the OpenASR runtime path satisfies placement, scheduler, and memory ownership
   constraints;
3. backend-level conformance has passed for the compiled plugin and target; and
4. reusable-graph and real-family evidence has passed where those stronger
   capabilities are requested.

`ggml_backend_dev_supports_op` is an appropriate no-allocation source for native
operator support. It is not, alone, evidence for persistent graph or real-model
correctness.

### Pack policy and lane capability

The pack and architecture determine:

- logits width and representation;
- decode policy;
- suppression, phrase-bias, timestamp, or probability consumers;
- state topology; and
- whether a compact token hint is semantically legal.

The physical execution lane determines:

- native operator support;
- provider/device identity;
- persistent graph capability;
- scheduler and capture behavior; and
- validated backend evidence.

The one shared planner combines those inputs before runtime-owner acquisition;
family code receives only the immutable output plan and reuse mode described
above.

Where graph topology or retained tensors differ, the output plan is part of the
runtime-owner/cache key. Current coverage is mixed:

- FireRed AED, Moonshine decoder, Granite, and Cohere unified-runtime keys
  already include output mode;
- Qwen and the FireRed LLM, FunASR Nano, MiMo, and MOSS wrappers keep the fused
  head resident and intentionally allow one owner to serve hint and non-hint
  requests; that remains legal only while one retained graph topology genuinely
  supports both plans;
- Cohere's split decoder is currently fixed to full logits; and
- SenseVoice can build full-logits or frame-token-id graphs while its cache key
  currently contains only pack and execution lane.

Migration must preserve deliberate topology unification rather than adding mode
to every key mechanically. If native `ARGMAX_FIRST` changes retained tensors,
output roots, allocation, or reusable graph topology, the resolved plan enters
the key. SenseVoice requires explicit correction or proof of one topology before
compact mode remains enabled. Native argmax capability is execution-lane
identity, not pack identity, and is never persisted as a model property.

### Correctness baseline

`FullLogits` (or complete scores for a non-token selector) means the model still
runs on the selected CUDA, Vulkan, HIP, or Metal FullDevice lane. Only the
selection row is read back for the family host oracle. It is not a CPU inference
fallback and does not weaken placement guarantees.

Any lane/tie-policy pair without complete compact-selection evidence uses the
complete-output plan. Unknown is not interpreted as supported.

### Native first-max path

For proven first-max lanes:

- CPU, CUDA, Vulkan, and HIP use native `GGML_OP_ARGMAX_FIRST`;
- Metal uses `FullLogits` until a native Metal implementation and all three
  conformance layers pass;
- family code requests a semantic first-max output rather than constructing a
  reverse gather; and
- the runtime validates that no request feature needs complete logits.

A last-max or other family oracle is a different output-plan request. It does
not reuse `NativeFirstMaxToken` and does not reinterpret ordinary parallel
`ARGMAX` as deterministic. XASR therefore retains host last-max selection until
a corresponding native lane capability is explicitly implemented and proven.
MiMo RVQ likewise retains host first-max code selection unless its compact
selector passes the same semantic contract.

After all consumers migrate, delete:

- reverse-index static tensors;
- reverse-index uploads;
- `GET_ROWS` vocabulary reversal;
- transpose/cont operations used only for reversal;
- reversed-token remapping helpers;
- reverse-index memory quotes; and
- source tests that preserve the obsolete path.

## Discriminating validation before final attribution

The implementation adds evidence and does not assume the confirmed tie bug
explains every symptom. The historical M1 q4 run diverged at the aggregate
`subsample_out` tap and therefore remains insufficient for Windows CUDA fp16
attribution. The current test-only twin exposes the ordered stem sequence
`mel_4d -> conv raw -> bias -> ReLU -> layout/cont/flatten -> output matmul ->
subsample_out`; a concrete seam is emitted only when every prerequisite tap is
present. The M1 q4/JFK rerun in
`gpu-decode-correctness-evidence/firered-encoder-stem-m1-q4-jfk.json` recorded all
12 taps and classifies the first CPU/Metal checksum difference as
`subsample_input` at `mel_4d`. That moves the investigation before the first
convolution; it does not prove whether input upload, view/readback, or another
backend boundary produced the checksum difference, and it still does not
project to CUDA.

### Same-graph dual output

Build one graph with one logits producer and two marked outputs:

1. the complete current logits row; and
2. native `ARGMAX_FIRST` from that exact row.

After one execution:

- compute host first-max from the returned logits;
- compare it with the device token;
- record the top two values and their margin;
- include explicit tie fixtures; and
- repeat with changing inputs to prove the scalar output refreshes.

This isolates selector semantics and output materialization from graph-to-graph
numerical differences. It is a diagnostic graph only. Marking a second output
can change ggml allocation and liveness enough to hide a stale-output defect, so
dual-output success never authorizes the production compact path. Authorization
comes from cases C and D below: the production-shape native-only graph is
executed repeatedly and compared with the host oracle from an independent
full-logits run.

### Fresh/reuse four-quadrant matrix

Use two independent runtime instances so that a fresh run cannot contaminate the
reusable runtime's KV arena.

| Case | Decoder graph | Selection | Question answered |
|---|---|---|---|
| A | fresh rebuild each step | complete logits/scores + family host oracle | baseline decoder correctness |
| B | reusable graph | complete logits/scores + family host oracle | reusable inputs/KV/output correctness |
| C | fresh rebuild each step | native compact selector for the declared tie policy | selector and scalar output correctness |
| D | reusable graph | native compact selector for the declared tie policy | combined persistent graph and compact output correctness |

For current first-max seq2seq consumers, cases C and D use native
`ARGMAX_FIRST`. XASR does not enter C or D until a native last-max capability
exists and passes its own oracle; it stays on complete logits in A/B. MiMo RVQ
uses the same matrix with complete scores and its declared first-max code oracle.

Interpretation:

- A correct, B wrong: reusable input, KV, mask, row update, or output-refresh
  defect.
- A and B correct, C and D wrong: native selector or compact output defect.
- A wrong from step 0 relative to CPU: encoder, cross-KV, decoder math, or another
  provider-kernel defect, not reusable KV.
- A-C correct, D wrong: interaction between persistent graph execution and
  compact output.
- All four agree on a bad CPU-relative sequence: inspect encoder and cross-KV.

### Encoder/decoder split probes

For the same short fixture and verified pack, compare:

- CPU encoder -> CPU decoder;
- CUDA encoder -> CPU decoder;
- CPU encoder -> CUDA fresh decoder;
- CUDA encoder -> CUDA fresh decoder; and
- CUDA encoder -> CUDA reusable decoder.

Record:

- encoder row shapes and checksums;
- selected layer-tap tolerances when a checksum diverges;
- cross-K/V checksums;
- per-step full-logits hashes;
- host and device selected token IDs;
- reusable row indices, positions, and mask hashes; and
- graph rebuild events when cross-frame topology changes.

If CUDA encoder -> CPU decoder already diverges, investigate FireRed's
relative-position attention, depthwise convolution, layout, and readback before
changing the decoder selector. If CPU encoder -> CUDA fresh decoder diverges,
focus on cross-KV upload or decoder kernels.

### Quantization order

Use fp16 to locate the first divergence. Once the implementation and short fixture
are frozen, repeat fp16, q8_0, and q4_k.

- all tiers diverge at the same stage: favor shared selector, reusable state, or
  non-quantized operator causes;
- only a quantized tier diverges: investigate its matmul/type/layout path; and
- only fp16 diverges: investigate f16-specific weights, KV, convolution, or
  embedding paths.

Do not start a full WER/CER sweep until short-audio per-step correctness is
closed.

## Three-layer conformance gate

No single layer substitutes for another.

### Layer 1: backend operator

On every actual backend claiming native first-max support, test:

- unique maximum;
- exact tie with first-max oracle;
- all values equal;
- negative values;
- the defined non-finite-value policy;
- single and multiple rows;
- the real FireRed vocabulary width;
- changing inputs across repeated execution; and
- fail-closed rejection of unsupported type, layout, or shape.

Run the tests on real hardware and the final compiled plugin. A software Vulkan
implementation is useful additional coverage but not proof for a physical
Vulkan device.

### Layer 2: reusable graph

For each enabled provider and graph shape:

- assert that reuse is actually active, not silently rebuilt;
- compare fresh and reusable full logits at every step;
- compare the device selector with the family-declared host oracle from the same
  logits/scores;
- execute repeatedly with changing token, row, position, and mask inputs;
- prove scalar and full-logits outputs refresh;
- validate in-place KV writes and subsequent reads;
- change cross-frame or logical topology and require rebuild; and
- validate the explicit scheduler and capture policy.

### Layer 3: real model family

Use:

- the release candidate binary and plugin;
- the staging signed catalog;
- a verified real `.oasr` pack;
- an isolated, newly created `OPENASR_HOME`;
- a fixed and hashed public short-audio fixture; and
- a CPU full-logits oracle generated by the same source revision.

For cold and same-process reuse runs, require:

- exact provider/device and FullDevice placement;
- no silent candidate or CPU fallback;
- per-step token trace parity;
- stop reason parity;
- degenerate-loop result parity;
- final transcript parity;
- output-plan and reuse-mode evidence; and
- bounded, artifact-bound receipts.

Close every advertised family/provider combination. Unsupported or untested means
not activatable for that family on that provider. It does not mean "shared ggml
path, expected to work."

A synthetic operator test remains necessary but insufficient. The prior Qwen HIP
incident demonstrates why only a real model exercises the complete operator and
shape composition.

## Release and audit contract

### Current gaps

The GPU optimization commit `d8fb8af8c` changed 113 files, added roughly 20,000
lines, and touched 15 model modules. The available release gates did not require
a real-pack correctness matrix for those affected families and providers.

Current evidence has different scopes:

- family regression pulls real packs and compares transcripts, but its execution
  is CPU-only;
- several model audit forms explicitly mark CUDA, Vulkan, or HIP as untested or
  deferred;
- the audit-form parser checks document structure and unfinished placeholders,
  not the semantic evidence in each backend cell;
- the Qwen GPU parity workflow is a manual raw diagnostic, not a release gate,
  and has no completed passing historical run at this evidence baseline;
- backend hardware evidence can prove artifact identity, fresh-process
  determinism, FullDevice placement, and no CPU fallback for its selected
  workload, but not every family or token path;
- Vulkan lavapipe smoke is software and synthetic; and
- scheduled and release-event family regression runs are post-publication
  monitoring; the reusable pre-publication CPU contract now blocks the release,
  while GPU correctness remains an exact-target activation gate.

These facts explain how CPU correctness, plugin packaging, and placement could
all appear healthy while FireRed CUDA token correctness remained unproved.

### Three gate classes

Keep these result classes separate:

| Gate | Proves | Does not prove |
|---|---|---|
| Build and packaging | build, ABI, signature, hashes, archive contents | runtime placement or model correctness |
| Placement and resource smoke | selected device, FullDevice compute, no fallback, resource observations | token or transcript correctness |
| Token and transcript correctness | real model behavior against the oracle | packaging completeness unless artifact-bound |

An `Activated` provider requires all applicable classes. The core release may
publish signed provider bytes as `PublishedInert` after build/packaging and the
CPU family gate; inert bytes are not runtime-selectable.

### Required release and capability-activation DAG

```text
build immutable release candidates
-> sign and attest candidates
-> generate and sign a staging catalog with every provider PublishedInert
-> run the release-candidate CPU family and packaging/CDN gates
-> obtain human release approval
-> deploy the PublishedInert catalog and publish those exact release bytes
-> run real-hardware backend evidence on the exact public bytes
-> run reusable-graph, family token-parity, and Desktop plugin-switch matrices
-> validate and bind both evidence classes to exact target and backend id
-> obtain separate human backend-scoped activation approval
-> sign and deploy a new Activated catalog epoch
-> run monitoring and retain one-way revocation
```

Publishing immutable `PublishedInert` artifacts is not capability activation.
The neutral runtime rejects those providers in Auto and explicit selection, so
post-publication hardware testing can bind the exact distributed bytes without
making an unproved lane usable. Do not activate first and treat release-event
regression as the blocker: qualification and explicit activation are separate
fail-closed transitions. Scheduled and release-event family jobs remain
monitoring and do not grant GPU authority.

### Generated evidence matrix

Generate the required matrix from the architecture inventory, public model
catalog, and backend catalog rather than maintaining an unrelated handwritten
family list. At minimum, include:

```text
public family and runtime topology
x advertised provider/placement
x representative shipped weight types needed to cover distinct kernels
x cold and same-process reuse
```

A provider can be built without being advertised for a family. Missing real
correctness evidence narrows activation support; it does not silently waive the
gate.

### Audit semantics

For any family/provider lane advertised as Auto-eligible or explicitly
selectable:

- `Untested`, `Deferred`, missing, or stale evidence blocks activation;
- a CPU result cannot populate a GPU cell;
- a placement receipt cannot populate a token-correctness cell;
- an audit form cannot grandfather a production execution path with no evidence;
- evidence includes artifact, pack, fixture, provider/device, driver, execution
  plan, and result identity; and
- the backend activation transition consumes both hardware and matrix receipts
  before a provider becomes publicly selectable;
- execution policy generates only family/provider placements whose correctness
  cell is approved; and
- memory pressure may choose among approved placements of the same model but
  cannot turn an unproved GPU failure into silent CPU execution.

### Desktop plugin-switch E2E

The product gate must exercise:

1. download the real release candidate plugin and vendor runtime;
2. verify signature, hashes, ABI, and target compatibility;
3. install with atomic selection state;
4. restart the daemon;
5. verify the actual provider and physical device;
6. transcribe a real verified pack;
7. restart the application and daemon again and prove selection persists; and
8. inject failures and prove the previous working backend is preserved.

A UI "ready" state cannot precede backend verification and the required product
smoke.

## Execution receipts

Every request using a compact device token or a reusable graph should expose a
bounded local receipt containing:

- requested execution intent;
- selected provider, stable device identity, and placement;
- resolved output plan, including complete logits/scores or the semantic native
  compact selector and declared family oracle;
- the capability evidence revision used to derive that plan;
- scheduler and capture modes;
- fresh or reusable graph mode;
- graph rebuild reason when topology changes;
- typed candidate failures and fallback chain; and
- token-trace or logits-trace artifact identity when running a conformance gate.

Production receipts must remain privacy-safe and bounded. They must not include
raw audio, model weights, secrets, or unnecessary local paths. Runtime policy
uses typed facts, never parsed receipt text.

Extend the existing [short-audio receipt](short-audio-receipt.md) schema and
artifact-binding path for these correctness records. Do not create a third JSON
receipt format or another policy authority. The extension must preserve the
existing receipt's versioning, deterministic identity, redaction, and validation
boundary while adding output-plan, token-trace, reuse, and lane evidence.

## Interaction with runtime ownership and model activation

GPU output correctness and runtime ownership are related but separate contracts.
They should not land as one unreviewable change.

The required sequencing is:

1. determine the first divergence with the evidence probes;
2. define shared lane capabilities and output-plan derivation;
3. replace the selector and migrate all consumers;
4. validate and enable provider/family cells; and
5. then implement the broader runtime ownership and atomic model-activation
   design against this corrected execution contract.

The ownership design may continue through review while this work proceeds, but
its implementation must not freeze the old assumption that a generic GPU-class
lane implies reusable-KV and compact-output correctness.

The implementation deepens the existing `GgmlBackendCapabilities` and
`ResolvedFamilyRuntimeInput` seam, following `GgmlNativeGqaCapability`. It must
not create a parallel capability registry, a second family-to-provider table, or
six family-facing evidence types. The shared planner resolves lane evidence,
family tie/decode policy, and request logits consumers exactly once.

The integration seam is:

```text
verified pack and family policy
+ exact execution lane and typed capabilities
-> immutable output plan
-> canonical runtime owner acquisition/materialization
-> graph execution and request receipt
```

The native first-max capability belongs to the physical execution lane, not the
pack. The pack determines output shape and whether its decode policy may consume
a token hint. The resulting output mode belongs in an owner/cache key whenever
it changes graph topology or retained tensors.

Activation failure must preserve the previous durable selection and active
runtime. This document does not redefine that transaction; it requires the
activation transaction to stage and attest the selected output plan before
publication.

## Migration plan

### Phase 0: evidence without semantic claims

Probe entry (diagnostic only):
`crates/openasr-core/src/ggml_runtime/decode_conformance.rs`. Dual-output
success does not authorize a production compact path.

1. Add same-graph dual-output support for diagnostic conformance builds only;
   never use that graph as production authorization evidence.
2. Add fresh/reuse four-quadrant probes using independent runtimes.
3. Add encoder/decoder split checks and bounded per-step receipts by extending
   the short-audio receipt path.
4. Reproduce FireRed fp16 on a short Windows CUDA fixture.
5. Identify the first divergent value and classify it as selector, reusable
   graph, encoder/cross-KV, or provider kernel.
6. Inventory XASR and MiMo RVQ ordinary device argmax against their current host
   tie policies.

Exit criterion: the field failure has a specific first divergence. The confirmed
tie bug remains a mandatory fix regardless of that classification.

### Phase 1: safe output planning

1. Resolve compact-token or compact-code capability in the one shared planner
   before owner acquisition.
2. Default every unproved lane/tie-policy pair to `FullLogits` or complete scores.
3. Preserve the selected GPU FullDevice execution and placement evidence.
4. Keep output plan in actor and cache identity where topology differs; explicitly
   close SenseVoice's full-logits/frame-token graph key.
5. Add tests proving every logits consumer forces `FullLogits`.
6. Keep XASR's last-max and MiMo RVQ's first-max host semantics unchanged.

Exit criterion: no unproved lane enters compact token selection.

### Phase 2: native first-max capability

1. Expose typed native first-max capability from the actual selected backend via
   `GgmlBackendCapabilities` / `ResolvedFamilyRuntimeInput`.
2. Use native `ARGMAX_FIRST` for proven CPU, CUDA, Vulkan, and HIP lanes.
3. Keep Metal on `FullLogits` until native support and evidence land; migrate the
   currently reachable Qwen-shaped Metal fused path rather than grandfather it.
4. Run backend-op and persistent-output tests on actual hardware.
5. Test HIP in its current capture-on configuration. If capture prevents the
   persistent contract from passing, disable capture explicitly for that lane or
   fix it before compact selection is enabled; do not inherit CUDA's capture-off
   evidence.

Exit criterion: every enabled compact-selection lane is same-graph equivalent to
the family-declared host oracle. For first-max consumers, the selector is native
`ARGMAX_FIRST`; other tie policies remain on complete outputs until their own
native capability is proven.

### Phase 3: migrate every consumer

Migrate FireRed AED, Cohere, Moonshine, Granite, Qwen, and every Qwen-shaped
wrapper. Preserve family policy checks for full-logits consumers. Bring XASR and
MiMo RVQ under the same semantic device-selection contract without changing
their host tie policies. Then delete the reverse helper, tensors, uploads,
remapping, and memory quotes, and correct the obsolete `cpu_graph.rs` comment
that describes ordinary native argmax as universally last-max.

Exit criterion: repository search finds no production reverse-selector path and
no parallel family/provider table.

### Phase 4: family and provider closure

Run the real-pack matrix for CPU, Metal, CUDA, Vulkan, and HIP according to
advertised support. Cover cold and reuse paths and the shipped weight types that
select distinct kernels.

Exit criterion: every enabled family/provider cell has current three-layer
evidence. Untested cells are not activatable.

### Phase 5: release and activation enforcement

1. Publish GPU provider artifacts only as `PublishedInert`; the ordinary runtime
   must reject them in both Auto and explicit selection.
2. Run exact-target hardware and correctness qualification only against the
   immutable public release bytes.
3. Make both evidence gates dependencies of the explicit backend activation
   transition, not of inert artifact publication.
4. Correct audit parsing and remove evidence-free grandfather paths.
5. Turn the CPU family regression into a release-candidate blocker.
6. Update model-family lifecycle and release documentation to record the
   separation between publication and activation.
7. Add desktop plugin-switch E2E before activation.
8. Retain scheduled and release-event jobs only as monitoring.

Exit criterion: no public provider capability can become selectable without the
required bound receipts, while unqualified release bytes remain inert.

### Phase 6: continue ownership and activation migration

Rebase the broader runtime-ownership implementation onto the corrected
capability and output-plan contract. Ensure staged owner attestation exercises
the selected plan and preserves the previous active runtime on failure.

## Validation matrix

### Short-audio correctness

For each affected topology and declared tie policy:

- CPU fresh complete logits/scores and the family host oracle;
- selected GPU fresh complete logits/scores;
- selected GPU reusable complete logits/scores where reuse applies;
- selected GPU fresh native compact selection;
- selected GPU reusable native compact selection where reuse applies;
- cold process and warm same-process rerun; and
- fp16 first, then q8_0 and q4_k where applicable.

Compare each device-selected token or code with its family host oracle before
evaluating final text. Record top-2 margins so a near-tie can be separated from
a gross logits divergence.

### Platform coverage

| Platform/provider | Required status before compact device selection |
|---|---|
| x86/ARM CPU | native operator and family host-oracle parity |
| Apple Metal | target state is complete logits until a semantic native selector and all gates pass; migrate the current Qwen-shaped reverse exception |
| NVIDIA CUDA | semantic operator, persistent graph, real-pack family parity; capture is currently off |
| physical Vulkan | semantic operator, persistent graph, real-pack family parity |
| AMD HIP/ROCm | semantic operator, current capture-on persistent graph, Qwen-shaped and other advertised family parity |

A single representative device may approve a target matrix only for behavior
that is genuinely target-invariant and cryptographically bound to all approved
artifacts. Family correctness cannot be projected from XASR to FireRed or from
one decoder topology to another.

### Negative coverage

Tests must fail closed when:

- the native operation for the requested family tie policy is absent;
- the selected output shape or type is unsupported;
- the request needs full logits;
- scheduler or the current capture mode lacks evidence;
- persistent inputs or outputs do not refresh;
- graph topology changes without rebuild;
- provider/device identity changes under a cached owner;
- a candidate falls back without a typed policy authorization; or
- required release evidence is missing, stale, or bound to another artifact.

## Work estimate and change boundaries

At the evidence baseline:

- the reverse selector appears in approximately eight Rust files;
- reusable compact-token paths appear in approximately thirteen Rust files;
- XASR ordinary device top1, MiMo RVQ device argmax, and SenseVoice output-plan
  cache identity extend the correctness surface beyond those counts; and
- affected work also includes shared capability code, tests, hardware scripts,
  short-audio receipt evolution, audit parsing, lifecycle/release policy,
  workflows, and desktop plugin-switch E2E.

The earlier 20-30-file estimate describes only the minimum core selector/reuse
migration. It is too narrow for the full contract. Plan separate reviewed
batches rather than one fixed file-count promise:

1. evidence probes, safe output planning, shared native selector, cache identity,
   and all device-side discrete-selection consumers;
2. provider hardware and real-family evidence, including HIP capture-on and
   Metal fallback/native support decisions; and
3. release/audit enforcement plus desktop product E2E across the open-core and
   app repositories.

A core-only first batch may remain near the original range, but the complete
program will exceed it. Hardware availability and release-policy integration are
likely to dominate elapsed time. If the four-quadrant probe identifies a FireRed
encoder, cross-KV, or reusable-KV defect in addition to compact selection, add a
focused family/runtime batch rather than expanding the shared selector change
with an unproved workaround.

The change should remain separate from the broad runtime-ownership migration,
although the latter must consume the resulting capability and output-plan
contract.

## Rejected approaches

### FireRed-only fallback

Disabling device top1 only for FireRed leaves the confirmed shared tie defect in
Cohere, Moonshine, Granite, Qwen, and Qwen-shaped families. It also preserves the
invalid abstraction for future families.

### Raising the repeat threshold

The repeat guard is correctly reporting a bad sequence. Raising or removing it
converts bounded corruption into a longer corrupted decode.

### Treating Dolphin as the GPU oracle

Dolphin exercises a different decode topology and does not validate
autoregressive compact selection or reusable KV.

### Provider-name allowlists in family code

A CUDA/Vulkan name match proves neither concrete operation support nor persistent
graph correctness. It drifts as backends and hardware evolve.

### Backend-specific reverse allowlists

Keeping reverse gather only where ordinary argmax currently appears last-max
would replace one false universal assumption with a backend behavior table.
Metal/CUDA/HIP/Vulkan would continue to diverge, and a kernel reduction change
could silently invalidate the table. The long-term abstraction is a native,
semantic tie-policy capability.

### Grandfathering Metal reverse gather

Metal's current reverse + stable-last implementation happens to produce
first-max, including in reachable Qwen-shaped fused paths. It is not a
publishable migration exception. Phase 1 moves every current Metal reverse
consumer to complete logits; diagnostic comparison may inspect the old path,
but no independently tested reverse result authorizes production. Metal compact
selection returns only after a native semantic operator passes the three-layer
gate.

### Permanent full-logits-only execution

Complete logits are the correct unproved-lane baseline, but making them the
permanent solution leaves significant avoidable transfers for large vocabularies
such as Qwen. Compact selection remains a valid performance goal after its
semantic operator and persistent graph are proven.

### A synthetic operator test as the only gate

It cannot exercise family graph composition, actual dimensions, KV reuse, masks,
quantized projections, or output-plan policy. It is one necessary layer.

### Final-transcript-only parity

Different token paths can converge to similar text, and short text may not enter
reuse. Per-step evidence locates the first divergence and prevents false passes.

### Release first, regression later

A post-publication failure cannot serve as a publication blocker.

### Bundling with the ownership redesign

Combining output correctness, backend capabilities, all runtime owners, and
atomic activation in one cutover would be difficult to review and would obscure
which contract corrected the field failure.

## Adversarial review questions

An independent review should try to refute this design by answering:

1. Is ordinary `ARGMAX` correctly described as a backend reduction behavior
   without a portable tie contract, rather than as universally first- or
   last-max?
2. Does any production reverse-selector consumer, indirect Qwen-shaped wrapper,
   ordinary XASR device token selector, or MiMo RVQ selector remain outside the
   blast-radius inventory?
3. Can any affected family legally consume device top1 while suppression, phrase
   bias, timestamps, probabilities, or debug logits are active?
4. Could the field repetition occur without any maximum tie, and does the
   four-quadrant matrix locate that alternative first divergence?
5. Can same-graph dual output perturb allocation or graph execution enough to hide
   the original defect?
6. Are fresh and reusable comparisons performed on independent KV state?
7. Does the proposed capability source confuse `supports_op` with proven
   persistent graph or real-model behavior?
8. Can Metal retain a correct, validated reverse selector temporarily without
   preserving the invalid cross-backend abstraction?
9. Does the current default-on HIP executable-graph capture alter input/output
   refresh semantics, and is compact/reusable execution disabled until that
   production configuration is proven?
10. Can a runtime owner be acquired under one output plan and reused under
    another due to an incomplete key, particularly SenseVoice or a Qwen-shaped
    fused owner?
11. Does any release evidence claim family correctness based only on placement,
    deterministic output, or another model family?
12. Can an audit cell remain `Untested` or `Deferred` while the corresponding lane
    is Auto-eligible or explicitly selectable?
13. Does the release DAG contain any path that activates the public catalog before
    correctness receipts are complete?
14. Can desktop plugin installation report success before the daemon has loaded,
    selected, and transcribed with the backend?
15. Would removing reverse gather regress a backend that lacks native first-max,
    and is the fallback explicit and tested?
16. Does the work estimate omit XASR/RVQ, SenseVoice cache identity, HIP capture,
    short-audio receipt evolution, generated bindings, audit/lifecycle policy, a
    release workflow, or desktop product E2E?
17. Is any proposed receipt unsafe for local privacy or used as an error-string
    policy side channel?
18. Does the sequencing preserve the previous active model and runtime when
    output-plan attestation fails?

## Acceptance criteria

The work is complete only when all of the following are true:

1. The field failure has a demonstrated first divergent stage on a representative
   short Windows CUDA fixture.
2. The confirmed first-max tie defect is fixed independently of the field
   attribution.
3. All production reverse-selector calls, tensors, remapping, and memory quotes
   are removed.
4. Every compact device-selected token or code matches its declared family host
   oracle under a typed lane capability; first-max consumers use native
   `ARGMAX_FIRST`.
5. Every lane lacking the required semantic capability uses complete
   logits/scores without changing the selected model backend or placement.
6. Device selection never bypasses a request feature that consumes logits or
   silently changes XASR/MiMo/SenseVoice tie or output semantics.
7. Fresh and reusable per-step logits and tokens agree on every enabled lane.
8. Persistent inputs, outputs, KV state, and topology rebuild behavior have
   explicit conformance evidence.
9. Every affected public family has real-pack correctness evidence for every
   advertised provider and required weight-kernel class.
10. CPU, Metal, CUDA, physical Vulkan, and HIP support are represented by actual
    evidence or by explicit non-activation.
11. Backend-op, placement, and token-correctness evidence remain distinct and are
    bound to release artifacts.
12. Family and provider correctness is a capability-activation blocker;
    publishing inert artifact bytes cannot make the provider selectable.
13. Desktop plugin switching passes install, restart, real-transcribe,
    persistence, and rollback E2E.
14. Request receipts identify provider/device, placement, output plan, reuse mode,
    and typed fallback without exposing sensitive data.
15. Runtime ownership and model activation consume the corrected output-plan
    contract and preserve the previous active runtime on failed activation.
16. No family-specific provider table, silent CPU fallback, alternate greedy
    loop, or evidence-free compatibility escape remains.
