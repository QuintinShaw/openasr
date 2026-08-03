# Decoder state and native memory planning

OpenASR separates model semantics, native physical footprint, concurrent
admission, and product fallback. No layer is allowed to infer or silently
rewrite another layer's facts.

## Invariants

1. The same request envelope has the same token/state shape on every machine.
   Memory pressure may select another approved execution placement, but it may
   not shorten the chunk, change the model, or change state precision.
2. A model position ceiling (RoPE/context/positional table) validates a demand.
   It is never used as an allocation request unless the runtime can genuinely
   address every position during one legal invocation.
3. Logical and stable shapes come from the same family-owned integer oracles
   used by the real frontend, prompt builder, and decode loop. There is no
   empirical `1.2x`/`1.5x` margin on an already-proven bound.
4. Independent state axes stay independent. Self-KV positions and cross-KV
   positions are not added together; their physical allocations coexist and
   are accounted as separate claims.
5. Native buffer bytes are quoted by the selected backend. Logical tensor
   payload is not relabelled as physical VRAM/RAM commitment.
6. Every OpenASR allocation is admitted against one process-wide broker for
   the physical memory domain it consumes. A native owner retains the broker
   lease until the native buffer is actually destroyed.
7. A family semantic invocation limit is independent of both memory pressure
   and the encoder's activation-scaling class. Fixed-window frontends are
   sliced before dispatch and reject an oversized direct call, so they never
   silently trim audio.

## Layer 1: model-semantic state topology

Decoder invocation coordinates are modality-specific under one generic
contract. `InvocationEnvelope` / `InvocationShapeInput` describe audio using
integer sample rate/count and sequence concurrency. `TokenInvocationEnvelope`
/ `TokenInvocationShapeInput` describe text-only decoders using prompt,
reusable-prefix, generation, and correlated total-position bounds. Text tokens
are never disguised as audio samples. Padding that is physically passed to a
model is included. Variable prompt content remains family-owned: the topology
obtains its exact upper bound from the same request-aware prompt builder used
by execution. The current invocation is kept separate from the stable
envelope.

Each decoder family implements the single required
`DecoderStateTopology::demands(scope)` oracle. `ExactInvocation` asks for the
current logical shape and `StableEnvelope` asks for the reusable capacity
proof. Every implementation must handle both scopes explicitly and reuse the
family's production shape and decode-budget functions. There is intentionally
no default that maps a stable envelope to one synthetic "maximum invocation":
that would silently assume monotonicity across correlated limits, padding
discontinuities, and batching buckets. The oracle returns one `StateDemand`
per independently-sized persistent stream. A stable state ID is the primary
key; `StateKind` is a diagnostic/semantic classification and may be shared by
multiple streams.

For a session envelope `W`, the general solution is the pointwise maximum of
the family-owned physical demand over every legal invocation in that envelope:

```text
K*_stream(W) = max { K_stream(x) | x is legal under W }
```

This is the unique smallest safe reusable capacity: any smaller value fails for
the invocation attaining the maximum, while any larger value is unused under
the declared contract. A monotone family may implement its `StableEnvelope`
branch by evaluating its maximum audio shape. A family with correlated limits,
padding discontinuities, or batching buckets supplies a different proof in
that branch; the generic planner never assumes monotonicity on its behalf.

Context legality and physical cache occupancy are deliberately different
coordinates. For the shared greedy schedule, a request remains semantically
legal only when

```text
P + G <= C_model
```

The first step writes all `P` prompt rows and samples the first output token.
Each of the remaining `G - 1` steps writes the preceding sampled token; the
final sampled token is returned but never fed back. Therefore the unique
minimum physical self-KV span for a causal-prefix request is

```text
K_self = P_max(samples, frontend, prompt contract)
       + G_max(samples, request decode policy, context remainder)
       - 1
```

The subtraction applies only to this proven schedule's self-KV writes. It must
never be used to enlarge the semantic invocation envelope, and it does not
apply to cross-KV, reusable prefix state, non-autoregressive state, or a future
decode schedule that feeds the final token back.

Checking only `K_self <= C_model` is unsound: `P + G = C_model + 1` can still
produce `K_self = C_model` because of the unwritten final sample. The shared
causal helper therefore validates `P + G <= C_model` first and derives the
physical `P + G - 1` span only after semantic legality is proven.

For an encoder-decoder family, self and cross state are separate:

```text
R_self  = S_max(initial decoder tokens, maximum generated tokens)
R_cross = E_max(samples, exact encoder/frontend shape)
```

`DecoderStatePlan` joins the current logical demand with the smallest proven
stable demand covering the session envelope. It fails closed when IDs differ,
the invocation lies outside the envelope, a reserve does not cover logical
state, arithmetic overflows, or a family/model cap is exceeded.

Families without token-scaled persistent state affirmatively return
`NoPersistentState`; absence of a plan is never interpreted as that result.
Every architecture descriptor also declares an
`OpenAsrDecoderStateTopology`: no persistent state, causal self-attention KV,
encoder-decoder self + cross-attention KV, or an explicit family-defined
multi-stream topology. Runtime materialization and CI compare that semantic
declaration with the executor contract. Each planned contract additionally
declares the complete stable ID + `StateKind` stream schema; dispatch rejects a
plan with a missing, extra, renamed, or misclassified stream. A newly added
causal/seq2seq family therefore cannot silently opt out, omit cross state, or
satisfy the check with an arbitrary non-empty plan.

Sequence concurrency is an allocation-owner property, not a synonym for serve
batch size. If one arena truly owns `N` simultaneous sequences, the invocation
contract carries `N`: positions remain per sequence while bytes scale by `N`.
The current serve path keeps one semantic plan per job, so each job correctly
declares one sequence. When the batch materializer builds one shared
`N`-sequence native arena, `N` is an explicit tensor dimension and the backend
quote/broker transaction admits that complete physical allocation atomically;
the batch-width guard also prices a per-slot contribution. Setting every job's
semantic envelope to `N` as well would double count the same concurrency.

Built-in architecture coverage is materialized and cross-checked by
`every_builtin_executor_declares_its_decoder_state_topology`. The family-owned
topologies currently cover:

| Family | Persistent streams | Bound source |
|---|---|---|
| Cohere | self + cross KV | request-aware prompt/decode budget + encoder frames |
| Whisper | self + cross KV | prompt/carry wrapper (including post-encoder LID prefix outcomes) + 256-token decode cap + aligned encoder window |
| Qwen | self KV | chunked audio encoder + prompt/carry + context clamp |
| Moonshine | self + cross KV | BOS + full `C-1` runtime decode budget + convolutional encoder length |
| FireRed-AED | self + cross KV | encoder-length decode budget + positional-table caps |
| FireRed-LLM | self KV | fbank/conv/adapter shape + fixed decode budget |
| FunASR Nano | self KV | fbank/LFR/adaptor shape + fixed decode budget |
| MiMo ASR | self KV | resampler/STFT/conv/RVQ grouping + fixed decode budget |
| MOSS-TD | self KV | chunk/merge/marker topology + proportional decode budget |
| Granite Speech | self KV | QFormer output + request prompt + fixed decode budget |
| Hy-MT2 auxiliary | scratch self KV + reusable prefix KV | correlated total cap + independent prefix cap |

`OpenAsrInvocationSpan` separately declares duration-only runtime limits.
For example, Whisper is bounded at its exact 30-second frontend window and
MOSS-TD is bounded by the product-approved 60-second invocation envelope.
The shared long-form policy applies these limits identically on CPU and GPU;
the topology/runtime check remains the direct-call backstop.

## Layer 2: native physical footprint

The optional ggml backend-memory ABI quotes concrete requests such as buffers,
host imports, transfers, scheduler-owned graph storage, and backend-private
allocator growth. Each quote reports:

- physical domain identity and fresh free/total observation;
- incremental peak and retained commitment;
- exact, upper-bound, or provisional confidence;
- a quote token used by the matching native allocation transaction.

The quote generation is an epoch only for provider state that can invalidate
the quote's layout/cost derivation. It is not a hash of live used/free bytes.
Capacity comes from the separately fetched fresh statistics; binding the epoch
to unrelated process allocations creates false stale failures without closing
the unavoidable race after any snapshot. CUDA and Metal v1 request-shape
quotes are independent of live free bytes, so they use a stable provider epoch;
allocation races remain covered by provisional reconciliation and typed OOM
fallback. Vulkan uses the same rule. Its buffer quote enumerates every memory
type the real retrying allocator may select and succeeds only when all of them
map to one broker domain. A configured device-local-to-system-memory fallback
therefore cannot silently move an allocation outside the domain that was
admitted; an ambiguous quote fails closed so the execution policy can choose an
explicit hybrid or CPU candidate.

Context tensor allocation mirrors ggml's real tensor-order packing and buffer
type maximum size. This matters for backends such as Vulkan, where a multi-GB
weight context becomes several bounded native buffers rather than one buffer
whose size equals the sum of tensor payloads.

Scheduler graph allocation uses a frozen native plan. Engine-controlled
buffer/transfer requests and backend-private high-water requests are admitted
as separate transaction classes; they are never combined into a value with
ambiguous ownership.

CPU scheduling helpers participate in the same contract. In particular, BLAS
quotes and transactionally reserves its reusable quantized-matrix conversion
workspace in the host/pageable domain; selecting `CpuOnly` preserves these
CPU-class helpers instead of silently changing the family's validated kernels.

## Layer 3: process-wide admission

`DeviceMemoryBrokerSet` is injected through one `NativeExecutionServices` root
per process and keyed by physical memory domain:

- discrete device heap;
- system memory shared by CPU and unified-memory accelerators.

A discrete accelerator without a proven canonical physical identity is not
given a backend-local substitute key: admission fails closed and execution
policy may try another approved candidate. This prevents CUDA/HIP/Vulkan
views of one physical card from receiving independent budgets.

All domains required by one transaction are reserved atomically. Concurrent
sessions therefore cannot each pass against the same unreserved bytes.

The footprint is phase-aware:

```text
candidate_peak(domain) = max over execution phases(
    sum of incremental commitments alive in that phase and domain
)
```

Weights and session-resident state remain live in their declared phases;
mutually exclusive encoder and decoder transient workspaces are not naively
summed. Existing committed allocations are owned by their existing leases and
only incremental commitment is charged to a new transaction.

Admission is also lifetime-aware across transactions. OpenASR does not pretend
that a complete future model/session footprint is knowable before ggml has
materialized every graph shape. Each real native allocation is quoted and
admitted just in time; previously retained weights/state remain charged by
their owner leases, while a released encoder workspace refunds its lease before
a decoder workspace is quoted. A complete frozen multi-owner plan may use the
phase formula above atomically. Otherwise the live broker ledger plus RAII
lifetimes is the authoritative peak proof. Neither path replaces typed
allocation-failure fallback for external memory races.

Exact and proven upper-bound quotes commit directly after successful native
allocation. Provisional quotes require a fresh post-allocation observation and
reconciliation. A partial backend-private failure that cannot prove release is
quarantined rather than optimistically refunded.

A provisional transaction holds its physical domain exclusively until that
reconciliation completes. This is intentional, not a throughput heuristic:
without a provider oracle that attributes backend-private deltas to individual
concurrent transactions, admitting two such allocations would make either
refund unsound. Exact and conservative-upper quotes remain concurrent; a future
provider with attributable commitment may remove only its own provisional gate.

Rust-owned persistent state uses the same transaction shape, with one important
accounting distinction. Before construction, each family declares a provisional
quote for engine-requested heap capacity from its tensor shapes and target host
storage. Construction uses fallible reservations; afterward,
`Vec::capacity * size_of::<T>()` measures the requested retained capacity. This number is not
allocator usable-size or physical RSS, so reconciliation uses the greater of
the provisional shape quote and measured capacity: it neither shrinks a proven
shape bound nor rejects legal allocator rounding above the estimate. A fresh
host snapshot and policy headroom cover allocator metadata, size classes,
fragmentation, and unrelated host pressure. All nested reservations in that
execution attempt share one cohort identity, so they may enter their own
provisional domain gate while unrelated candidates remain atomically excluded.
Only a future platform allocator usable-size oracle may safely enable more exact
post-build shrink reconciliation. The allocation and lease move into one owner
whose field order drops memory before refunding the lease. Semantic plans
themselves never own reservations.

Reusable host materializations are retained through a cache-neutral
`AdmittedHostObject<T>` handle. Its shared LRU has independent entry-count and
committed-requested-byte ceilings; a cold miss obtains its family quote before
materialization, while a hit neither requotes nor reallocates. An object larger
than the cache byte ceiling may execute but is not retained. Eviction drops only
the cache's `Arc` clone, so an in-flight request keeps the object and its lease
alive. Per-key slots preserve cold-build single-flight, and a clear or eviction
during a build cannot republish the detached slot.

Small auxiliary packs that are parsed after deferred admission use an immutable
anonymous snapshot of the already-open file generation. Snapshot copying writes
directly into the anonymous mapping and hashes each copied block, avoiding an
additional pack-sized `Vec`. The policy retains the original preflight mapping
for semantics-preserving candidate fallback, so materialization overlaps that
mapping, the anonymous snapshot, and the parser/runtime construction peak. The
safe bound is therefore `2S + M`, where `S` is the pack byte length and `M` is
the materializer's own construction peak. The snapshot content id must equal the
preflight key before any graph is built. Multi-GiB ASR packs remain mmap-backed
and do not take this auxiliary-only copy path.

Every runtime GGUF parse uses the same engine-wide C parser limits for tensor
count, metadata count, individual string bytes, array elements, and total header
bytes. The parser validates declared counts before reserving descriptor storage
and reports allocation exhaustion as a typed capacity failure. A generic
`GgufRuntimeSourcePreflight` binds the already-open generation, bounded metadata,
and bounded tensor index; validation, quote, and materialization reuse that one
provenance unit rather than reopening or reparsing the path. Auxiliary OADP
admission additionally derives a tighter pack-specific quote from fixed-header
counts, ABI-reported native container sizes, and string wire multipliers. Normal
multi-GiB model payloads remain mmap-backed; the parser limits constrain only
header structure, never tensor payload bytes or model context length.

Recording-scoped auxiliary pipelines reserve transient system memory with the
same phase rule. Their provider contract exposes integer shape oracles for the
actual sliding-window count, activity-frame count, bounded inference
concurrency, padded tail, and model/frontend payload. The planner also charges
the window/result headers and retained outputs that coexist with each phase,
then reserves only the maximum of segmentation, VAD, embedding, clustering,
reconstruction, and centroid phases. It does not multiply duration by a nominal
floating-point FPS and it does not sum buffers whose lifetimes cannot overlap.

## Layer 4: execution policy

The policy resolver orders semantics-preserving candidates only:

- `auto`: full-device candidates, then supported hybrid candidates, then CPU;
- explicit accelerated: full-device/hybrid only, never pure CPU;
- CPU: CPU only;
- exact/provider/vendor-constrained targets: only matching proven devices.

Only typed capacity, device-unavailable, or device-lost failures authorize the
next candidate. Model, input, format, cancellation, and decoding failures do
not. String matching is not a fallback protocol.

Offline dispatch may retry the complete transaction. Streaming construction
and warm-up may retry before any audio is accepted. After the first sample is
accepted, automatic replay on another candidate is forbidden.

Window reduction, model switching, and state quantization are separate product
choices and are not hidden inside the planner or fallback resolver.

## Text auxiliary decoders

Realtime Hy-MT2 uses the same topology planner with a text invocation
contract. The product clause segmenter and decoder capacity share one
construction-time `max_emitted_clause_chars` value. Its default hard bound is
40 Unicode scalars. Hy-MT2 uses byte-level BPE, whose successful output has at
most one initial symbol per UTF-8 byte and whose merges only reduce token
count, so source text has the proof-backed bound

```text
T_source <= 4 * max_clause_chars = 160
```

The fixed prompt content is also bounded by its UTF-8 bytes (rather than
assuming separately tokenized prefix counts compose across a pre-tokenizer
boundary), while role/generation markers are counted exactly. The tokenizer
enforces `encoded_tokens <= UTF-8 bytes` at runtime and fails closed if a pack
or future implementation violates the proof.

One translation candidate declares two simultaneous stable state streams:

```text
R_decode_scratch = P_max + G_max - 1
R_session_prefix = Prefix_max
```

Decode scratch owns host history plus the reusable resident graph. Each
streaming translation session owns a separate host prefix arena. Clause
switches, finalization, and failed prefix prefill clear written positions and
metadata in place; they never free/reallocate the admitted arena. Candidate
construction allocates and warms both streams before it may be published.

## Product envelope

The default long-form target is 30 seconds and VAD may extend toward a natural
boundary while keeping the fully padded executor input at or below 60 seconds.
The state envelope uses that same fed-window ceiling; it does not add padding a
second time. This 30/60 contract must pass CER, DER, and seam A/B gates before
release; memory planning proves capacity for the selected contract but does
not choose its quality trade-off.

For the shipped MOSS-TD decoder (28 layers, 8 KV heads, head dimension 128,
fp16 resident K+V), one position is 114,688 bytes:

| Shape | Positions | Resident KV |
|---|---:|---:|
| current 30s invocation | 1,289 | 141.0 MiB |
| stable 60s product envelope | 2,366 | 258.8 MiB |
| former unconditional arena | 8,192 | 896.0 MiB |

The reusable session therefore saves about 637 MiB (71%) versus the former
arena while still covering every legal slice. This is a KV result, not an OOM
guarantee: weights, backend-private workspace, other sessions, and external
GPU users remain part of physical admission and typed fallback.

## Boundary behavior

- A one-second request still uses the real frontend/prompt oracle and the
  family's positive minimum decode budget. It never rounds down to an empty
  cache.
- A configured 300-second invocation is accepted only by a family whose
  semantic span and model context prove it legal. A 30/60 product family fails
  before allocation; the memory planner never silently shortens it.
- A small learned/RoPE ceiling (for example Whisper's 448-position semantic
  context) remains the legality cap. Physical self-KV is still derived from
  the actual prompt plus the family's generation budget; only a runtime that
  genuinely permits generation through the full remaining context (Moonshine,
  for example) requires `C-1` rows under the current greedy schedule.
- Checked integer arithmetic makes zero rates/strides, overflow, an exhausted
  prompt context, or an empty generation budget fail closed.
- Percentage margins are not added to a proven token bound. Architectural
  padding/alignment is represented explicitly by a conservative integer proof;
  allocator/driver uncertainty belongs to physical quotes and policy headroom.

## Comparison with simpler allocation policies

| Policy | Reallocation | Waste bound | Multi-family correctness | Physical admission |
|---|---|---|---|---|
| `transcribe.cpp`-style user `n_ctx` | once per session | up to the user/model context gap | caller must understand each model's topology | allocator failure only |
| CrispASR-style request resize | may grow on requests | near zero after exact resize | request-local shapes can be exact | repeated GPU growth can add latency and fragmentation |
| OpenASR topology envelope | once per reusable owner/envelope | current demand to proven session maximum | self/cross/prefix streams and family oracles stay explicit | process-wide atomic byte admission plus typed fallback |

The OpenASR policy combines stable reusable buffers with exact working-set
reasoning. The product envelope, rather than the model's mathematical context
or the current audio length, supplies the stable preallocation boundary.

## Family onboarding checklist

For every new decoder family:

1. expose fail-closed model metadata for every relevant state geometry/cap;
2. provide a count-only integer frontend/encoder shape oracle shared with the
   real tensor construction path;
3. provide one decode-budget oracle shared by topology and `DecodeConfig`;
4. implement both `ExactInvocation` and `StableEnvelope` branches of the one
   `demands(scope)` oracle, returning stable-ID `StateDemand` streams with
   explicit logical/resident bytes and a family-owned maximum proof;
5. declare the precise `OpenAsrDecoderStateTopology` and pass the
   executor/descriptor startup cross-check;
6. allocate reusable state from `DecoderStatePlan`, validating the actual
   logical and resident shapes;
7. declare provider/placement capabilities truthfully;
8. declare any semantic single-invocation duration limit and make both shared
   slicing and the direct runtime enforce it;
9. route every native allocation through backend quote and broker admission;
10. test one-second, default, maximum, overflow, small model-cap, and concurrent
   reservation cases.
