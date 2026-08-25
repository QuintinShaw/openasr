# Runtime ownership and atomic model activation

Status: software contract implemented on the integration branch; release
acceptance remains blocked on the real-host provider matrix described below.

This document defines the cross-family contract for resident runtime ownership,
physical-memory planning, candidate materialization, and durable model
activation. It complements:

- [Decoder state and native memory planning](decoder-state-memory-planning.md),
  which defines semantic state shapes and physical admission;
- [Runtime source preflight](runtime-source-preflight.md), which defines the
  one-open immutable source proof; and
- [Model-family lifecycle](model-family-lifecycle.md), which defines the
  architecture inventory and shared/family boundary.

The live Rust inventory, backend ABI, and tests remain the implementation
authority. This document now serves as both the governing contract and the
release checklist for the transaction from a verified pack and execution intent
to one admitted resident owner and one durable active model selection.

## Implementation checkpoint

The software-side migration is complete for the current inventory:

- architecture descriptors build the resident topology consumed by candidate
  quote/reserve and default activation;
- broker reservations, native allocations, host allocations, pack residency,
  thread-affine backend handles, checkout actors, forced-aligner assets, and
  serve-batch semantic children publish scope-local typed receipts;
- live lease reconciliation compares the NES scope rather than a process-global
  event ring, and event-history truncation cannot invalidate a complete live
  owner table;
- `CandidateActivationTransaction` is the shared typestate path; ordinary
  publication rejection rolls back, while only typed may-have-mutated failures
  quarantine;
- `ActiveRuntimeSlot` separates durable requested intent from attested process
  state, serializes activation against new sessions, publishes only after V2,
  and startup reactivation validates V2 without minting a new generation;
- `default_selection` V2 stores architecture plus a reversible, path-free
  execution-intent wire value, uses the existing same-directory atomic writer,
  and has per-attempt injection at every pre-commit and replace boundary; and
- Desktop delegates activation to the daemon and vendors the generated HTTP
  contract. It does not calculate capacity or persist a competing selection.

This does **not** convert missing hardware into success. Local-dev HIP
(capture-on) and physical Vulkan (`vk_caps` exact cell, capture unsupported)
decode evidence.v1 now exist on one RX 9060 XT. They are still not a
production-catalog or Authenticode cell.

The remaining product hole on that host was not a missing family matrix.
`serve --model-pack` used to set `requested_path` and listen while `active`
stayed empty, because boot reactivation silently returned unless the path was
already an `InstalledModelStore` object named by durable V2. Loose `.oasr`
files are no longer a second runtime: `--model-pack` fail-closes before bind
unless that store+V2 pair exists. The working admit path is catalog-gated
`pull --from` then `POST /v1/models/default` (the existing activation
transaction). That path now binds `funasr-nano:q4` on the Vulkan-qualified
neutral host (`model_installed`/`model_resident`, HTTP JFK). A
`cold_warm_lifecycle` `openasr.runtime-ownership-evidence.v1` envelope for
that same cell now passes `__openasr-validate-ownership-evidence`
(diagnostic-not-release): baseline empty, cold=warm live owners after JFK,
idle-unload `now` returns to empty, lease matched. GRAPH_PRIVATE high-water
is one backend-owned lease on the cached GPU backend; a fresh-graph builder
must reuse that row instead of admitting a new zero-byte `native-memory-owner`
per compute. A real-host pressure-rollback envelope is UNAVAILABLE on this
close-out (ColdWarm is the accepted alternative). CUDA, production-signed
plugin activation, and the packaged product kernel-switch flow still require
release-bound real-host receipts. The gate must remain closed while those
cells are missing, stale, or unavailable.

## Accepted completion program

The remaining work completes the release and real-host portions of this
contract. Existing HTTP ownership snapshots and deterministic tests are strong
diagnostics, but they are not a substitute for the artifact-bound ownership,
pressure, capability, and product gates defined here.

### Orthogonal artifact and capability state

Publication is not a linear state enum. It is the product of two dimensions:

- artifact state is `compiled` or `published` and is proven by immutable release
  subjects, hashes, checksums, and attestations;
- capability state is `qualification_only`, `activatable`, or `revoked` and is
  derived from signed capability/catalog state plus complete exact-cell
  receipts.

Architecture inventory describes code and topology; it is not the mutable
publication authority.

A qualification manifest may contain only the release subject, binary/plugin/
vendor hashes, host ABI, provider target, attestation, and immutable download
locations. It carries no activation modes and never enters ordinary runtime
candidate generation.

Each exact cell has one collision-free release asset named
`openasr-<version>-qualification-<provider-target>.json` (bundled Vulkan uses
its already provider-qualified `vulkan-windows-*` target without repeating the
provider). Its detached signature binds the identical basename at
`https://dl.openasr.org/core/v<version>/`; a mirror may transport those bytes,
but cannot change their canonical signed identity. CI compiles only unsigned,
inert manifests after the successful provenance action emits its exact Sigstore
bundle. The production catalog seed remains local; one fail-closed maintainer
command signs every cell on the draft release, uploads only detached signatures,
then re-downloads and verifies every published pair.
The CI upload paths, local signer, and final `draft -> published` transition
share one atomically-created, nonce-owned draft-release lock asset. A stale lock
fails release completeness; no manifest may be replaced concurrently with
signature creation or publication. The finalizer reconstructs the exact cell
set from backend packs and re-verifies every manifest/signature, referenced byte,
and provenance subject while holding that lock. The shared Rust verifier,
not only the generator script, rechecks the release version, host artifact,
provider/target plugin name, content-addressed vendor name, provenance bundle,
and canonical CDN/GitHub URLs before either signing or qualification loading.

On Windows, `artifacts.binary` binds both the exact `openasr.exe` member and
the attested release ZIP that contains it. The ZIP also carries a signed
unpacked-byte total and canonical tree digest, so qualification rejects a
correct executable combined with changed or extra companion DLLs. Plugin DLLs
and vendor ZIPs use the same signed URL/hash discipline in a qualification-only
content namespace; none is installed into the ordinary backend store.

Before qualification succeeds, the ordinary capability-aware runtime catalog
contains no CUDA candidate compatible with the new host ABI. The explicit
qualification runner consumes the signed manifest in an isolated child process
using the exact final release bytes. It refuses the real user home, does not
write ordinary `active.json`, accepts no arbitrary plugin path, and exposes no
Auto or Explicit activation path.

The explicit `__openasr-qualify-backend` parent verifies and prepares the
manifest, then starts the same executable as a fresh
`__openasr-qualification-child` bound to the parent's manifest SHA. The child
re-verifies the production signature, exact host ABI, release-bundle tree,
artifact hashes, and attestations before loading anything. Windows file handles
deny write/delete sharing from the final rehash through provider execution;
directory components reject symlinks, junctions, and reparse points.

GitHub CLI is currently part of the qualification-host trusted computing base
for Sigstore bundle verification. The runner resolves `gh.exe` once to an
absolute non-linked file, holds a deny-write/delete handle, records its version
and SHA-256, passes the signed repository/workflow/source/predicate constraints,
and supplies the signed offline bundle. This is an explicit host-tool
assumption, not an OpenASR artifact correctness proof; replacing it with an
in-process Sigstore verifier may narrow the TCB later without changing the
manifest or capability contract.

### Old-client rejection

An unknown ordinary catalog field is not an old-client gate. The first
capability-aware release must:

1. bump `BACKEND_HOST_ABI_SCHEMA_VERSION` so previous stable clients hide or
   reject every new qualification artifact;
2. carry backend `min_cli_version` into `ResolvedCatalogBackendPull` and enforce
   it during resolve, prepare, and install;
3. set a later public CUDA entry's minimum version to the first
   capability-aware stable release; and
4. run the previous stable binary as a black box and prove the new entry cannot
   resolve, install, or activate.

After qualification, a public CUDA entry binds the new ABI, minimum CLI version,
release subject, artifact identity, and signed capability matrix digest/epoch.
`active.json` remains only an atomic pointer and never substitutes for the
current signed catalog or capability proof.

Legacy-ABI entries require signed tombstone/removal semantics. An offline client
cannot know a tombstone it has not downloaded; this limitation is documented and
does not justify forced networking or phone-home behavior.

### Typed exact-cell approval

`ExecutionCandidate` remains a pure device/placement type. Once the verified
pack, family, quant, topology, candidate, output plan, and reuse plan are known,
the shared `CapabilityApprovalResolver` performs an O(1) typed lookup against an
immutable, already verified in-process `CapabilityApprovalSnapshot` and returns
an `ApprovedExecutionCandidate`.

Only the approved type may enter activation, request dispatch, resident owner/
cache identity, or release-bound receipts. Family code cannot parse provider
names, catalogs, matrices, or approval records.

Approval is checked at four boundaries:

1. catalog resolve/prepare/install validates artifact publication and host
   compatibility;
2. plugin activation validates artifact-level capability proof;
3. daemon boot revalidates current signed catalog, epoch, matrix digest, and
   tombstones; and
4. model activation plus every request candidate generation validates the exact
   family/model/quant/topology/provider/target/plan/mode cell.

Approval epoch and matrix digest partition owner/cache identity. After a
tombstone or epoch change, an old owner cannot be checked out by a new session.
No network access, signature verification, or JSON parsing occurs on the request
hot path.

### Artifact-bound ownership evidence

`openasr.runtime-ownership-receipt.v1` remains the production diagnostic
snapshot for owner/resource/domain/lifecycle/completeness and ledger facts. Its
random redaction key belongs to one service root, so redacted identities are not
compared across daemon starts.

Formal release evidence wraps, rather than replaces, those snapshots:

```text
artifact-bound ownership evidence envelope
|- release / core / plugin / pack / catalog bindings
|- ordered phase list
|- daemon start identity per phase
|- request / activation receipt bindings
|- hashes of runtime-ownership-receipt.v1 snapshots
|- expected transition assertions
`- final result
```

The envelope is not an admission or policy authority. Cross-process continuity
uses pack SHA, artifact identity, phase order, and daemon start identity. Within
one process, request facts, doctor, and plugin activation attestation bind the
raw selected device to the HTTP snapshot's redacted lane identity without
publishing sensitive raw identifiers. The finalizer verifies every referenced
snapshot and receipt hash.

### Deterministic and real pressure gates

Both gates require a causal state transition.

The deterministic race requires:

```text
baseline forecast succeeds
-> broker/native facts change
-> activation reads fresh facts and rejects reserve
-> old durable/live runtime remains
-> staged owners release or quarantine correctly
-> ledger reconciliation matches
```

The Windows real-host gate uses a repository-owned helper that actually commits
and touches memory. It preserves absolute and proportional available-memory
safety floors, refuses to cross either floor, has a hard timeout, runs inside a
Job Object with kill-on-close, cleans up on parent death, continuously checks the
floor, never locks pages, and releases all memory after every terminal path.

A passing real-host sequence requires:

```text
same pack/exact lane is admissible and old runtime is active
-> pressure helper becomes ready
-> native observation crosses the rejection threshold
-> the same candidate fails activation on a fresh observation
-> old durable selection remains unchanged
-> old live runtime completes a real transcription
-> staged owners clean up and ledger reconciliation matches
-> helper exits and available memory/observation recover
```

If the baseline already fails, the helper does not cross the boundary, crossing
would violate a safety floor, the failure identity changes, or cleanup/recovery
is incomplete, the result is `UNAVAILABLE`, `BLOCKED_BY_HARNESS`, or `FAIL`, not
PASS.

### Revocation and safe restart

A downloaded signed tombstone supersedes cached approval. New activations and
sessions cannot enter the revoked lane, and old owners cannot be checked out.
An in-progress native call is not interrupted by unloading its DLL. The daemon
drains it, blocks new work, and restarts onto a bundled approved backend. The
offline client continues to use its last locally verified signed state until a
new state is explicitly obtained.

### Release sequence

The release pipeline has two separately authorized gates:

1. finish the implementation and local/HIP/physical-Vulkan evidence;
2. publish the formal capability-aware release and inert CUDA qualification
   assets only after explicit user approval;
3. run the qualification runner on the external CUDA host using those exact
   published bytes;
4. bind passing receipts to exact capability cells;
5. publish a signed capability/catalog epoch only after a second explicit user
   approval; and
6. activate those cells in the already-published runtime version without
   creating a second binary release.

If qualification fails, the assets stay inert and the code fix enters a later
version.

### Completion workstreams

The coordinated program consists of six workstreams:

1. exact-route Layer 1, capture-aware Layer 2, production-shape four-quadrant,
   and observed graph lifecycle shared by HIP, physical Vulkan, and CUDA;
2. formal real-family `ShortAudioReceipt evidence.v1` production with complete
   logits/token traces and matrix binding;
3. physical Vulkan artifact-bound hardware evidence;
4. packaged Tauri product E2E through public IPC, the production kernel-switch
   transaction, `DaemonSupervisor`, real transcription, persistence, and
   rollback;
5. artifact-publication/capability-activation gates, qualification, exact-cell
   approval, old-client rejection, and signed revocation; and
6. the ownership evidence envelope, finalizer consumption, deterministic race,
   and safe real-host pressure/rollback harness.

## Executive decision

The Windows FireRed failure that motivated this design is not a broker arithmetic
bug. The broker correctly rejected a 5,092,073,216-byte system-memory allocation
when both the policy remainder and the observed budget were smaller. The product
failure is broader:

1. resident runtime ownership was expressed by several family-specific cache
   and actor shapes, so the complete number and lifetime of physical owners was
   not represented by one enforceable contract;
2. the runtime could quote and admit each native allocation safely, but the
   model selection path could not evaluate and stage the complete selected
   route before changing durable state; and
3. desktop and server persisted the requested default model before the new
   runtime had been successfully admitted, materialized, attested, and
   published.

The remedy is not a FireRed special case, a larger memory margin, a global cache,
or a static model-size estimate. OpenASR now uses one standard owner protocol
and one activation transaction while preserving model semantics, backend
memory semantics, thread affinity, and bounded parallelism.

This is a deepening of three existing modules plus one schema upgrade, not four
parallel subsystems:

1. `NativeExecutionServices` and its actor/cache primitives expose one enumerable
   resident-owner protocol;
2. the architecture inventory exposes one backend-neutral resident-footprint
   facet, while `NativeMemoryAdmissionPlan` remains the provider-specific
   physical expansion and admission authority;
3. the existing candidate attempt/journal is the one transaction used by
   offline, streaming, warm-up, auxiliary, and activation paths; and
4. `default_selection` owns a versioned schema and activation commit
   protocol instead of gaining another persistence authority.

Every migration must include a deletion test: once a component enters the new
protocol, source/inventory audits must prove that its old publication, retry,
owner, and persistence path no longer exists.

## Scope

This design covers:

- all live ASR architecture rows and every persistent or request-scoped auxiliary
  runtime, including owners not represented by an ASR inventory row;
- offline transcription, stateful streaming, startup warm-up, serve-batch, and
  explicit default-model activation;
- CPU, Metal, Vulkan, CUDA, and HIP providers when compiled and enumerated;
- host-imported, file-backed, copied backend, graph-private, scheduler, KV, and
  other persistent or transient allocations;
- one `NativeExecutionServices` root and its process-wide physical-memory
  broker; and
- server and desktop state transitions around the open-core runtime.

It does not require one binary to contain all five providers. Every provider
compiled into and enumerated by a host must follow the same contracts.

This design does not change these product policies:

- native execution remains fail closed;
- an exact or accelerated-only request does not silently append CPU;
- memory pressure does not shorten audio, change the model, reduce state
  precision, or alter model semantics;
- an installed model may remain installed when it is not currently activatable;
- the runtime does not silently select another default model; and
- a quote is not a reservation and cannot replace allocation-time admission.

## Incident evidence and epistemic boundary

### Confirmed facts

The relevant failure is recorded in the supplied logs:

- supplied daemon diagnostic, lines 770-775
- supplied Desktop diagnostic, lines 1199-1203

The system-memory broker reported:

```text
requested          =  5,092,073,216 B
committed          = 11,488,973,972 B
pending            =              0 B
unreclaimable      =              0 B
policy ceiling     = 15,406,611,046 B
policy remainder   =  3,917,637,074 B
observed ceiling   =  4,461,342,720 B
```

The arithmetic is exact:

```text
15,406,611,046 - 11,488,973,972 = 3,917,637,074
5,092,073,216 > 3,917,637,074
5,092,073,216 > 4,461,342,720
```

Both the ownership-policy gate and live-observation gate therefore rejected the
allocation. Under the same observation snapshot, clearing only the broker's
committed ledger would not make the request admissible. Releasing old owners may
also increase real host availability, so the log does not prove that a fresh
process would still fail.

The resource names `pack-weight-buffer-chunk-0` through `chunk-5` describe six
simultaneously live native buffers whose total is 5,092,073,216 bytes. They do
not show six copies of a 5-GB pack. Context chunking reproduces the backend's
actual maximum-buffer, alignment, tensor-order, and allocation-size constraints
in `crates/openasr-core/src/ggml_runtime/cpu_graph.rs`.

The failure occurred while constructing the FireRed encoder's loaded weight
context. It is not evidence that the 5,070-second input directly caused the
weight allocation. Long-form input has separate session and graph costs, but the
limiting resource in this receipt is the pack weight buffer.

### Strong hypothesis, not yet a finding

The 11,488,973,972-byte committed total is an aggregate system-memory account.
The current failure does not identify the contributing pack, component, actor,
thread, backend, or allocation kind. It therefore cannot prove that two FireRed
weight copies account for the committed total.

The duplicate-owner hypothesis is nevertheless plausible because:

- `GgmlLoadedWeightContext` is shared by a thread-local weak cache keyed by
  execution scope, mmap identity, and backend address
  (`cpu_graph.rs:1481-1494`, `cpu_graph.rs:1893-1929`);
- a copied backend binding cannot be shared merely by recognizing the same
  content id; it is a real backend allocation owned by a thread-affine context;
- streaming warm-up and offline execution may materialize owners on different
  pinned threads or backend instances;
- several family executors permit a bounded number of actors for one key; and
- the unified GPU runtime path does not cover every family, provider, placement,
  and streaming path.

The architecture must make such multiplicity explicit and charge it correctly,
whether the FireRed reproduction eventually confirms or refutes this specific
hypothesis.

### Additional facts that the current logs cannot resolve

Before declaring the incident fully explained, a reproduction must determine:

1. which owner receipts compose the 11,488,973,972-byte committed account;
2. why the native system-memory policy basis at failure was materially below the
   machine's reported total physical RAM, including raw native
   `total/budget/free`, observation kind, and selected native domain;
3. whether the two server boot sequences in the log were overlapping daemon
   processes, and how much external pressure each process contributed; and
4. whether startup warm-up and the offline request used the same pack content,
   exact execution lane, backend instance, and loaded binding identity.

These are required validation questions, not prerequisites for fixing the
known persistence-first activation defect.

## Current coverage

### ASR family coverage

The live architecture inventory in `crates/openasr-core/src/arch/mod.rs` contains
16 ASR families. All participate in shared execution services, admission,
streaming completeness, owner eviction, and default-model warm-up.

Fourteen families use the pack-wide `GgmlLoadedWeightContext` binding. The
binding prefers mmap host import and falls back to a copied backend weight buffer
when the selected backend cannot import the mapping:

| Runtime binding | Families |
|---|---|
| `GgmlLoadedWeightContext` | Cohere, Whisper, Qwen, Moonshine, FireRed AED, FireRed LLM, FunASR Nano, MiMo ASR, MOSS Transcribe-Diarize, SenseVoice, Wav2Vec2 CTC, Granite Speech, Parakeet CTC, Parakeet TDT |
| family-owned mixed loader | Dolphin, XASR Zipformer |

`GgmlLoadedWeightContext` is an ownership mechanism, not a synonym for a copied
buffer. CPU and Metal normally import a file-backed mapping. Vulkan, CUDA, and
HIP currently use copied weight buffers. Host import may fail and take the copy
fallback, so ownership is determined by the actual materialized backend path,
not by family or provider name alone.

Dolphin and XASR do not use the loaded-context cache, but their actor instances
can still retain family-specific host copies, backend graph state, and device
owners. No live family is outside the broader runtime ownership and activation
problem.

Cross-actor duplication is bounded by pool policy; this design addresses an
unrepresented multiplicity contract, not an observed unbounded leak.

The owner inventory must not stop at the 16 ASR rows or wrap only existing pool
types. Phase 0/1 must explicitly account for current shapes that can otherwise
escape:

- FireRed LLM split execution constructs request-scoped encoder and adapter
  owners while its decoder uses a checkout pool;
- the same executor also owns a distinct unified-GPU pool, so split and unified
  topologies require separate keys, limits, receipts, and eviction even when one
  request chooses only one topology;
- FireRed Stream-VAD has a process-level embedded model plus host session and
  `NativeExecutionServices` thread-pinned accelerated actors, despite not being
  one of the 16 ASR architecture rows; and
- every other auxiliary, request-scoped materializer, family-local cache, and
  split/unified pool must be discovered from construction sites and then made an
  inventory projection with no wildcard or `NotApplicable` escape for a real
  owner.

Serve-batch does not add a third physical-owner concurrency policy. It is a
serialized owner actor whose active batch slots are explicit session footprint.
Batch-width-specific decoder, KV, or graph runtimes retained inside that actor
are resident components of the same owner and must be priced; queue width,
`max_native_sessions`, and batch-slot memory may not remain implicit capacity.

### Provider coverage

| Provider | Normal weight path | Physical accounting requirements | Important uncertainty |
|---|---|---|---|
| CPU | file-backed host import; copied host buffer on fallback | system memory, file-backed residency, exact reusable workspace | separately opened mappings are conservatively distinct; copied fallback is thread/backend local |
| Metal | mapped file-backed buffer; unified-memory copy on fallback | CPU and accelerator share one system-memory policy domain | driver/command costs are opaque and require non-zero headroom |
| Vulkan | copied backend buffer | physical UUID and heap identity; `VK_EXT_memory_budget` observation | graph-private allocation can be provisional; missing budget evidence fails closed |
| CUDA | copied backend buffer | device-local or unified domain from native device facts | quote is provisional and backend-owned pool counters omit direct weight buffers |
| HIP | CUDA-derived copied backend buffer | device-local or unified domain from native device facts | provider accounting reliability remains explicitly less proven than CUDA |

The backend adapter, not model-family code, is the only layer allowed to
interpret native domain kind, physical identity, quote confidence, observation
confidence, allocation token, reconciliation, trim, and quarantine.

A single scalar `memory_bytes` cannot represent these semantics. In particular:

- file-backed policy residency is not the same as observed physical pressure;
- Metal's unified working-set budget is not dedicated VRAM;
- Vulkan heaps cannot be merged by provider name;
- CUDA/HIP pool counters cannot be presented as complete device ownership;
- exact, conservative-upper, and provisional claims have different commit
  protocols; and
- KV, copied weights, scheduler high-water, and opaque driver costs have
  different owners and lifetimes.

## Existing mechanisms to preserve

This proposal extends rather than replaces several existing correct mechanisms.

### Immutable source proof

`GgufRuntimeSourcePreflight` binds one already-open source generation, bounded
metadata, and tensor index. Validation, planning, and materialization must not
reopen the path. `VerifiedPack` remains the proof entering execution.

### Physical-domain broker

`DeviceMemoryBrokerSet` is process-wide and keyed by physical domain. It already
provides:

- atomic multi-domain reservation;
- policy and live-observation gates;
- reservation and committed-ledger accounting for claims already classified by
  the adapter as exact, conservative-upper, or provisional;
- candidate-exclusive reconciliation for unattributable provisional growth;
- owner-lifetime refund; and
- quarantine when release cannot be proven.

The broker is authoritative for physical admission. It must not learn model
families or product fallback policy.

### Native backend quote and reconcile

`NativeMemoryAdmissionPlan` joins fresh native statistics, quote tokens, domain
mapping, broker reservations, materialization, and post-allocation
reconciliation. It remains the provider-specific quote/reserve/reconcile seam;
not every native allocation is created by one Rust function today, and the
design must not overstate that narrower boundary.

For direct GPU execution, persistent model tensors must retain the existing
`GGML_BACKEND_BUFFER_USAGE_WEIGHTS` placement and compatibility checks. State or
compute buffers cannot be relabeled as weights to satisfy placement. This is a
weight-placement and graph-correctness gate, not a replacement for physical
memory admission.

### Architecture inventory

`OpenAsrArchitectureDescriptor` is the single family inventory. This proposal
must add required facets to that inventory rather than create a second family
registry or central family-id match.

### Typed execution policy

`ExecutionPolicyResolver` remains the source of ordered, semantics-preserving
candidates. Only typed capacity, device-unavailable, device-lost, or placement
failures may advance to another candidate. Error strings are not policy.

## Design invariants

1. Every resident native allocation has exactly one canonical owner protocol,
   one lifetime, and one physical-memory receipt.
2. Every owner is scoped to one `NativeExecutionServices` root. Only the physical
   broker is process-wide.
3. A runtime owner key includes exact content, source generation, representation,
   component, and execution-lane identity. A family id, path, or coarse `GPU`
   label is insufficient.
4. Thread-affine native contexts are constructed, used, and destroyed on their
   owner thread. Thread identity is enforced by the actor protocol, not copied
   into the reusable logical key.
5. Parallel owner multiplicity is explicit, bounded, and included in the
   footprint contract. Each simultaneous checkout owner has its own instance
   identity, receipt, lease, and drop responsibility.
6. Serve-batch is scheduling inside a serialized owner, not a third owner policy.
   Its retained batch-width runtimes are resident components; active member
   state and slots are session footprint.
7. Model code declares semantic topology, representation requirements, and
   concurrency. It does not estimate provider bytes or implement
   provider-specific admission.
8. Backend code expands semantic allocation intent into physical requests. It
   does not choose models, alter semantic envelopes, or persist product state.
9. For stateful streaming, candidate replay is allowed only before the first
   externally visible semantic output or decision. After that commit frontier,
   the lane is pinned and cross-lane replay/fallback is forbidden.
10. A model selection is not active until its candidate runtime has been
    admitted, materialized, attested, reconciled, durably recorded, and
    atomically published.
11. A failed activation leaves the previous durable selection and active runtime
    unchanged.
12. Static catalog RSS is descriptive product metadata, never authoritative
    admission data.
13. Advisory forecast never replaces the final race-safe reservation.
14. Receipts describe decisions but never become an error-string policy side
    channel.
15. After the planned-topology migration gate, production family code may
    materialize declared components on demand but may not allocate, retry,
    publish, or fall back through an unplanned family-local JIT path.
16. A persistent graph's raw backend and scheduler handles are covered by the
    same shared native lifetime owners as its runner/cache entry. Runner or
    thread-cache teardown cannot free those handles first, and scheduler
    replacement fails before mutation while any persistent session retains the
    old scheduler.

## Target architecture

The target deepens the existing architecture inventory,
`NativeExecutionServices` owner/cache layer, native admission plan, candidate
journal, and `default_selection` schema. The names below describe responsibilities,
not permission to create parallel registries, brokers, caches, or persistence
systems. Review may refine Rust spelling without weakening the boundaries.

### `RuntimeFootprintContract`

Each architecture descriptor must provide a runtime footprint facet that
constructs a declarative topology from:

```text
VerifiedPack
+ ExecutionCandidate
+ execution intent
+ request/session envelope
```

The contract contains:

#### `ResidentComponentSpec`

A stable description of each long-lived component:

- component id and variant;
- content identity;
- phases in which it remains live;
- sharing scope;
- thread-affinity requirement;
- concurrency policy;
- provider/placement compatibility; and
- dependencies on other resident components.

Examples include a pack-wide loaded binding, encoder runtime, decoder runtime,
shared scheduler, diarization auxiliary runtime, or family-owned immutable host
materialization.

#### `SessionFootprintSpec`

Request/session-scoped state derived from family-owned integer shape oracles:

- KV and cross-attention state;
- streaming state;
- decoder state;
- bounded batching/concurrency contribution; and
- retained versus phase-transient lifetime.

The existing decoder-state topology remains authoritative for decoder state. The
new contract references it rather than duplicating its formulas.

#### `NativeAllocationIntent` in the family facet

The family facet may declare only backend-neutral semantics:

- required representation class, such as persistent weights, mutable state,
  transfer, or graph workspace;
- component dependencies and phase lifetime;
- shape/session envelope;
- sharing and checkout multiplicity; and
- provider/placement compatibility already expressed by execution inventory.

It must not contain provider byte estimates, physical domains, alignment, buffer
chunks, native quote tokens, or CUDA/HIP/Vulkan/Metal/CPU branches.

#### `NativeAllocationSpec` in the ggml/backend adapter

Only the adapter expands a semantic intent for one selected lane into physical
allocation specifications. It derives:

- host import versus copied binding;
- native buffer chunks and alignment;
- physical domain identity;
- scheduler/graph-private claims;
- quote and observation confidence;
- fresh statistics and quote tokens; and
- reservation, reconciliation, trim, and quarantine requirements.

This is a deepening of the existing ggml runtime and
`NativeMemoryAdmissionPlan`, not an allocation table owned by each family.

#### `FootprintConfidence`

Each claim remains typed as exact, conservative upper, provisional, or unknown.
Unknown physical cost is not interpreted as zero and cannot pass admission.
Opaque provider costs require a proven non-zero domain headroom policy.

### `ResidentKey` and `CanonicalResidentOwner`

`ResidentKey` is the reusable logical identity and must include at least:

```text
pack content id
already-open source/mmap generation identity
architecture id
component id and variant
adapter/LoRA fingerprint
representation partition: host-neutral or device-owning
ExecutionLaneKey when device-owning
decoder/session resident span or capacity class
NativeExecutionServices scope id
```

A host-neutral object intentionally has no execution lane. A device-owning object
must have one. `ExecutionLaneKey` must retain provider, stable physical device
identity, placement, and graph backend. Different CUDA/HIP/Vulkan views,
different cards, or different placements do not share merely because they are
accelerators. The key retains the already-resolved full route so a worker can
reinstall exact provider/device selection without enumeration; cache equality
and hashing use the stable route cache identity and deliberately exclude only
registry ordinal.

A stateful streaming session is already pinned to one policy candidate, so its
session request carries a mandatory `ExecutionLaneKey`, not an optional hint.
Seq2seq and CTC drivers copy that same key into every partial and final frame's
execution context. They do not re-enumerate a device or omit the lane on a
direct family decode path. XASR uses the same key for actor-pool checkout,
owner-thread construction, and every later streaming operation, including
warm-up and reset; ambient thread-local device state never creates its key.

A bounded checkout additionally uses:

```text
OwnerInstanceKey = ResidentKey + checkout slot + instance generation
```

Every simultaneously live instance receives its own receipt, lease, health
state, and drop responsibility. Checkout slot is not folded into `ResidentKey`,
because doing so would confuse reusable logical identity with one physical pool
instance. Thread identity is in neither key; the actor protocol enforces
thread-affine construction, use, and destruction.

`CanonicalResidentOwner` owns, in drop-safe order:

1. native runtime, scheduler, graph, actor, `AdmittedHostObject`,
   `SystemMemoryOwner`, or family host object;
2. pack/content handles and `pack_weight_residency` file-backed residency
   handles;
3. host-memory owner leases;
4. device and backend-private reservations;
5. health state: healthy, poisoned, quarantined, or evicted; and
6. a receipt collector.

The owner protocol supports exactly two concurrency policies:

- **serialized actor:** one thread-affine runtime processes commands serially;
- **bounded exclusive checkout:** a declared maximum number of independent
  owners may execute concurrently, and every possible owner is priced.

Serve-batch uses the serialized-actor policy. Its owner thread may schedule many
member sessions and retain several batch-width runtime variants, but these are
respectively session footprint and resident components inside that owner. They
do not create a third physical owner protocol.

This standardizes keys, leases, publication, eviction, poison, and drop without
forcing all models into one actor implementation.

The resident registry belongs to its `NativeExecutionServices` root. A global
resident cache is forbidden because it would join unrelated CLI, server,
embedded, and test-host lifetimes. The process-wide broker remains global because
all roots consume the same physical memory.

### `CandidateActivationTransaction`

This deepens the existing candidate attempt/cache journal; it is not a second
runner beside it. Offline, streaming, warm-up, auxiliary preparation, and
default-model activation must use the same candidate transaction protocol:

1. **Prepare** constructs an immutable plan from the verified, already-open pack.
   It performs no native allocation.
2. **Resolve** obtains the ordered candidate list from
   `ExecutionPolicyResolver`. Exact and accelerated-only intents retain their
   existing fail-closed behavior.
3. **Quote and reserve** obtains fresh backend facts and atomically reserves all
   known physical domains. A frozen complete footprint may reserve its
   phase-aware peak; otherwise only components and checkout instances already
   declared by the prepared topology may use JIT owner admission, with existing
   live leases remaining authoritative.
4. **Materialize staged owners** creates canonical owners without publishing
   them to a shared registry.
5. **Attest and reconcile** runs the minimum legal warm graph or first required
   compute, verifies provider/placement, and reconciles provisional growth.
6. **Commit** publishes owners and cache entries only after every required check
   succeeds.
7. **Rollback or quarantine** destroys staged owners in reverse construction
   order. A may-have-mutated native failure quarantines rather than falsely
   refunding memory.
8. **Fallback** advances only for the typed failures authorized by execution
   policy.

A dry-run footprint query may reuse Prepare and Quote for product guidance. It
must report that it is advisory: no reservation exists and external/native state
may change. Activation always repeats fresh quote/reserve and remains the final
authority.

For stateful streaming, buffered input may be replayed on another typed candidate
only until the first externally visible semantic output or decision commits the
lane. Raw PCM arrival is not itself the frontier. After commit, replay and
cross-lane fallback are forbidden.

During migration, existing JIT allocation may remain only behind an explicit
tracked compatibility seam. The Phase 3 exit gate forbids production family code
from constructing any component, checkout instance, cache entry, retry, or
fallback that was absent from the prepared topology. Declared components may
still be materialized on demand for request-dependent shapes; “planned” does not
mean “eagerly allocate everything.”

### `DiagnosticReceipt`

Every activation attempt produces bounded, production-safe structured receipts:

- `PackPreflightReceipt`: content, architecture, route, and proof result;
- `CandidateReceipt`: candidate order, lane, capability basis, and typed result;
- `RuntimeFootprintReceipt`: component/resource id, allocation kind, domain,
  requested/peak/retained bytes, confidence, raw native observation, quote token
  generation, reuse/materialization, reconcile result, and quarantine;
- `ActivationReceipt`: selection generation, old/new identity, transaction
  stage, fallback chain, and commit/rollback outcome.

Receipt fields must be machine-readable and stable. They must not expose secrets,
raw model bytes, local audio metadata, or unnecessary local paths. The server may
retain a bounded ring for local diagnosis. UI and logs receive summaries; policy
uses typed values directly and never parses receipt text.

Owner receipts must make this query possible:

```text
For one physical-domain committed total, list every live owner and its
content/component/lane/allocation-kind contribution.
```

That query is required to turn the FireRed duplicate-owner hypothesis into a
confirmed or refuted finding.

## Atomic model activation

### Resolved baseline defect

The baseline server and desktop persisted requested selection before a path-only
runtime rebind. That sequence could leave durable selection, in-memory binding,
and UI state describing different models.

Production model routes no longer call the legacy path-only rebind. Set-default
enters the transaction below; deletion commits V2 `Unset` before a non-fallible
active-slot clear; Desktop is a daemon delegate. On startup, durable V2 is only
requested intent: the fresh `ActiveRuntimeSlot` remains unavailable until the
same transaction reverifies, reserves, warms, attests, reconciles, validates the
unchanged durable record, and publishes the process-local identity.

Resident readiness follows the same transaction boundary. The process-wide
idle tracker grants an exclusive unload claim only after one complete idle
epoch; session/activity enter and that claim are serialized, so a new request
cannot start after the reaper observed zero activity but before owner teardown.
Warm state is keyed by attested pack content identity plus the process-local
runtime generation, not by one global boolean or path. Activation holds an
activity guard across verification through live publication, advances the
generation immediately before candidate warmup, and does not advance it again
after the candidate has proved resident. Consequently candidate warmup cannot
make the old pack look resident, rollback cannot publish the candidate, and a
successful boot reactivation retains both its worker warm state and exact
`model_resident` receipt.

### `ModelActivationTransaction`

Server activation wraps the core candidate transaction:

1. acquire an activation barrier and reject or drain activity according to the
   existing session policy;
2. resolve and verify the installed pack to `VerifiedPack`;
3. stage a fully admitted, materialized, attested, and reconciled runtime through
   `CandidateActivationTransaction`;
4. prepare one versioned durable `ActiveModelSelectionV2` record;
5. atomically persist that record;
6. perform an in-memory publication designed to be non-fallible after durable
   commit; and
7. release the previous runtime only after publication.

`ActiveModelSelectionV2` is a versioned schema upgrade inside
`openasr_core::default_selection`, not a new selection module or parallel
authority. Its reader preserves the existing semantic result:

```text
Installed | NotInstalled | Unset
```

`NotInstalled` is a valid persisted user intent. It is not cleared, silently
replaced, or converted into automatic selection. The V2 record is the durable
source of truth and includes:

- selection generation;
- pack content identity and logical pull identity;
- architecture and quant preference;
- execution intent or exact route preference; and
- schema/version information required for recovery.

The existing `config.json` and default pointer may be written as compatibility
projections for one migration window. They must no longer be independent sources
of runtime truth.

Ordinary CLI resolution and the no-selection last-resort model lookup are
read-only and must never write V2. Model pull/install must not automatically set
the default. If CLI exposes an explicit activate/set-default operation while a
daemon owns the service root, it delegates to the daemon activation transaction;
it does not bypass admission or write the record directly.

Failure semantics are strict:

- failure before durable commit: publish nothing, preserve the old durable record
  and runtime, and release/quarantine staged owners correctly;
- durable write failure: do not publish the staged runtime;
- durable replacement uses the existing `atomic_file` primitive with one
  same-directory private staging file that is fully written and synced before
  replacement. On Windows the replacement path must retain
  `MoveFileExW(MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)` semantics;
- durable commit and in-memory publication are two recoverable states, not one
  magically uninterruptible instruction. If the process is killed after record
  replacement but before pointer exchange, the next process treats V2 only as
  the requested selection, reverifies, and reactivates it. It never claims that
  a previous process's runtime is still active;
- V2 defines a single-writer activation barrier, record validation/checksum,
  failpoints before and after replace, and rules to ignore or clean orphaned
  staging files. Startup must observe either the old complete record or the new
  complete record; best-effort parent-directory sync alone is not claimed as a
  cross-platform durability proof;
- UI, tray, or diagnostic rendering failure after successful activation does not
  roll back the core state transition; and
- an installed but non-admissible pack remains installed and selectable later
  when capacity or route changes.

The exact atomic persistence mechanism must follow the existing local-state
safety contract: private staging, sync as required, no partially visible record,
and startup recovery from an interrupted replacement.

## Layer ownership

### Open core

Open core owns all trusted and fail-closed logic:

- footprint contract and inventory facets;
- resident keys, canonical owner, and scoped owner registry;
- backend quote/admission integration;
- candidate transaction and typed fallback;
- receipt wire types and safe redaction;
- active-selection record schema, atomic persistence primitive, and migration
  reader; and
- conformance gates for every model and auxiliary runtime.

### Server

Server owns host orchestration only:

- activation barrier and session/realtime coordination;
- active runtime slot and non-fallible publication seam;
- activation command and error mapping;
- startup reactivation from the durable selection;
- bounded receipt storage and local diagnostic endpoint; and
- warm-up as a caller of the common candidate transaction.

The request-concurrency semaphore remains distinct from physical-memory
admission. It controls service QoS; the broker controls physical capacity.

### Desktop

Desktop owns product experience only:

- request activation from the daemon;
- wait for an `ActivationReceipt::Committed` result before displaying success;
- display installed-but-currently-unavailable state and the limiting resource;
- expose execution preference using daemon-authoritative provider/device data;
- retain compatibility for existing `auto`, `cpu`, and `accelerated` preferences;
  and
- never implement a parallel TypeScript memory formula or write default-model
  state directly.

The catalog's static RSS field may remain a coarse download/discovery hint. It
must not enable or certify runtime activation.

## Approaches explicitly rejected

### FireRed-only cleanup or retry

Deleting an old runtime and retrying may make one reproduction pass but does not
prove ownership, preserve active sessions, or close other families. It converts
a deterministic transaction into destructive trial and error.

### Raising the policy ceiling or reducing headroom

The incident also failed the observed-capacity gate. Weakening policy hides real
pressure and increases native OOM/device-loss risk without fixing duplicate
owners or persistence ordering.

### Global cache keyed only by content id

Two copied backend buffers with identical content are two real physical owners.
Charging one would oversell memory. A native context may also be thread-affine
and provider/device/placement-specific. Content identity is necessary but not a
complete resident key.

### One actor for every runtime

This would serialize models and components that safely support bounded parallel
execution. The shared abstraction is the owner protocol, not one scheduling
policy.

### Static catalog RSS or pack-size admission

Static bytes cannot represent backend allocation, zero-copy versus copy,
chunking, actor multiplicity, session state, graph high-water, UMA, driver
private costs, current owners, or external pressure.

### Dry-run quote as a guarantee

A quote does not reserve capacity. External processes and native allocators can
change after the snapshot. Advisory forecast improves UX; activation-time
reservation and reconciliation remain authoritative.

### Silent fallback to another model

Execution fallback may choose another approved placement of the same model.
Selecting a different model changes product semantics and requires explicit
user policy. It is not a memory-recovery mechanism.

### One large cutover with parallel old and new owners

Running both owner systems concurrently would distort the very memory accounting
being migrated. Migration may adapt old owner implementations behind the new
protocol, but each family/component has one authoritative owner path at a time.

## Migration plan

### Phase 0: evidence before behavior change

This phase changes no admission ceiling, fallback policy, owner lifetime, or
selection behavior.

1. Add production-safe owner/resource receipts to existing broker leases, loaded
   contexts, scheduler owners, host objects, actor pools, request-scoped
   materializers, and auxiliary owners.
2. Explicitly instrument FireRed LLM request encoder/adapter construction, split
   decoder and unified-GPU pools, FireRed Stream-VAD embedded/host/device owners,
   and serve-batch retained runtime variants.
3. Emit raw native observation kind and `total/budget/free` without changing
   policy.
4. Add daemon PID/start identity to lifecycle receipts.
5. Reproduce startup warm-up followed by offline transcription on one isolated
   service root and classify every committed owner.

Exit criterion: the 11.49-GB incident hypothesis can be confirmed or refuted from
owner receipts rather than aggregate arithmetic.

### Phase 1: contracts and adapters

1. Add the backend-neutral footprint facet to the existing architecture inventory,
   add the owner protocol/registry to `NativeExecutionServices`, extend existing
   receipts and `NativeMemoryAdmissionPlan`, and keep one candidate journal. Do
   not create parallel registries or admission systems.
2. Adapt existing cache and actor implementations behind the owner protocol.
3. Inventory every ASR, auxiliary, request-scoped, split/unified, host-object,
   and serve-batch owner. No wildcard or `NotApplicable` may hide an owner that
   allocates or retains memory.
4. Extend validation so every owner topology declares representation partition,
   dependencies, phase lifetime, and concurrency.
5. Run receipt-only shadow comparison against existing admission decisions.
6. For each component switched to the new protocol, delete its old publication
   path in the same change and add a source/inventory test proving it cannot
   return.

Exit criterion: every resident construction site is enumerable from the
inventory/audit, and shadow receipts account for every existing lease.

### Phase 2: representative owner migrations

Migrate one representative of each owner shape first:

1. file-backed host-import runtime;
2. thread-pinned copied-weight runtime;
3. bounded checkout pool;
4. graph-private/provisional accelerator runtime;
5. family-owned mixed loader;
6. request-scoped materializer beside a retained pool, using FireRed LLM split
   execution as the required case;
7. split and unified pools for one family; and
8. non-ASR-row auxiliary and serve-batch serialized owners.

Implementation note: “bounded checkout pool” means the existing
`AdmittedPinnedRuntimeActorCheckoutPool`; the short-lived parallel
`ResidentCheckoutPool` prototype was deleted rather than retained as a second
authority. Serve-batch runtime/session rows are typed `NoBrokerLease` semantic
children because their native/system-memory child owners publish the actual
brokered byte receipts; `Unknown` is reserved for negative tests and cannot
silently compare as zero. Forced-aligner prepared assets use
`SystemMemoryOwner`, while its audio/decoder/logits transients carry distinct
stage lanes.

For each migration, remove its previous direct publication path. Verify
construction/drop order, content-id eviction, idle unload, poison, quarantine,
and cancellation before migrating the remaining families.

Exit criterion: no migrated family has dual owner paths, and its physical owner
multiplicity matches the declared contract.

### Phase 3: common candidate transaction

1. Move offline, streaming, warm-up, auxiliary, and default-activation candidate
   loops to the deepened candidate transaction.
2. Preserve current execution-policy ordering, typed fallback semantics, and the
   stateful-streaming semantic-output commit frontier.
3. Make staged cache publication and rollback common.
4. Add advisory footprint query using the same plan/quote path without treating
   it as a reservation.
5. Remove production family-local allocation/retry/publication paths that can
   materialize an owner absent from the prepared topology. Request-dependent
   declared components may still materialize on demand.

Exit criterion: all candidate materialization enters one transaction protocol;
source audits reject handwritten retry/publication loops and any unplanned JIT
owner escape.

### Phase 4: atomic active-model state

1. Upgrade `default_selection` in place with the V2 record, atomic replacement,
   `Installed | NotInstalled | Unset` reader, legacy migration, crash failpoints,
   and recovery tests. There is never a parallel V2 persistence module.
2. Remove pull/install auto-default writes and add the CLI firewall before making
   V2 authoritative.
3. Introduce the server active-runtime slot and activation barrier.
4. Replace persistence-first server rebind with `ModelActivationTransaction`;
   current path-only rebind is not counted as materialization.
5. Make startup warm-up reactivate the durable V2 requested selection through the
   same transaction.
6. Retain legacy files only as compatibility projections from
   `default_selection`, never as independent writers.

Exit criterion: a failure at every pre-commit stage preserves both old durable
selection and old active runtime.

### Phase 5: desktop projection

1. Make the Tauri command a pure daemon activation delegate.
2. Remove desktop default-model persistence authority.
3. Render committed, rolled-back, unavailable, and fallback receipts.
4. Retain legacy execution-target values and add exact route only from
   daemon-authoritative device enumeration.

Exit criterion: desktop never reports a model active before the daemon commits
it, and no TypeScript capacity formula controls activation.

### Phase 6: delete compatibility paths

After the defined compatibility window and release gates:

- delete legacy persistence-first rebind;
- delete duplicate family candidate loops and owner wrappers;
- delete independent default-state writers;
- turn legacy state readers into explicit migration-only code, then remove them;
- strengthen inventory/source audits so regressions fail CI.

## Validation matrix

### Weight-free conformance

For every architecture descriptor:

- footprint facet is explicit and has no wildcard/default escape;
- resident component ids and variants are unique;
- provider/placement capabilities match execution inventory;
- owner concurrency is serialized or explicitly bounded;
- construction enters canonical owner and candidate transaction seams;
- bare paths, coarse GPU keys, family-local brokers, and direct cache publication
  fail source/inventory audit.

### Broker and backend simulation

Cover:

- exact reproduction of the incident numbers;
- dry-run leaves the ledger unchanged;
- forecast succeeds, capacity changes before activation, fresh activation
  correctly fails, and no durable selection or active pointer changes;
- final reservation repeats fresh facts and remains authoritative;
- multi-chunk context totals and per-chunk alignment;
- exact, upper-bound, provisional, residual, stale quote, and unavailable stats;
- post-allocation overrun and quarantine;
- file-backed same-mapping reuse and distinct-mapping conservative charge;
- host-import failure followed by copied-buffer fallback;
- same content/same lane reuse;
- same content/different provider, device, placement, or scope miss; and
- explicit bounded actor multiplicity.

### Lifecycle and concurrency

Cover:

- cold materialization and warm reuse;
- startup warm-up followed by offline and streaming requests;
- FireRed LLM split request encoder/adapter, decoder checkout, and unified-GPU
  owners receive distinct planned receipts with no untracked construction;
- FireRed Stream-VAD embedded, host-session, and accelerated actor owners are
  enumerable despite not being an ASR architecture row;
- serve-batch has one serialized physical owner policy, while retained
  batch-width runtimes and active slot/session state are separately priced;
- stateful streaming may replay before its first externally visible semantic
  output/decision and cannot replay or switch lane afterward;
- idle unload and content-id eviction;
- actor panic, cancellation, device loss, poison, and quarantine;
- a failed staged owner is never observable concurrently;
- owner-thread destruction precedes broker refund;
- two service roots share physical admission but not resident owners; and
- multiple daemon processes are visible as external pressure, not merged owner
  receipts.

### Activation failure injection

Inject failure at every stage:

- pack verification;
- candidate resolution;
- quote/stat observation;
- broker reservation;
- native materialization;
- first-compute attestation;
- reconciliation;
- V2 staging write and sync;
- atomic replacement;
- durable commit before in-memory pointer exchange; and
- pre-publication process restart.

Also cover:

- `Installed`, `NotInstalled`, and `Unset` V2 resolution;
- CLI read-only fallback, pull/install no-auto-default, and explicit
  daemon-delegated activation;
- same-directory replacement and crash failpoints immediately before/after the
  Windows replace operation;
- advisory forecast success followed by activation-time pressure rejection; and
- server HTTP, desktop Tauri, and desktop receipt-rendering projections.

Before durable commit, the old durable selection and active runtime must remain
unchanged. After a committed durable record and restart, startup must reverify and
reactivate it rather than claiming stale process memory. Orphan staging files
must be ignored or safely cleaned, and recovery must accept only a complete,
validated old or new record.

### Family and provider matrix

Run all 16 ASR families and persistent auxiliary families over every
inventory-declared provider/placement combination. Unsupported combinations must
be absent from candidate generation, not fail later and silently append CPU.

Real-host release gates must include at least:

- CPU-only host;
- Apple Silicon Metal;
- NVIDIA CUDA;
- AMD HIP/ROCm;
- discrete Vulkan; and
- integrated/UMA Vulkan.

Each real-host gate uses a real development `.oasr` pack and exercises cold,
warm, and pressure conditions. Performance measurements, when required, run in
an otherwise clean exclusive window; correctness builds and tests need not be
serialized merely because they touch backend code.

For the original class of failure, the minimum real-host sequence is:

```text
single daemon with PID receipt
→ activate one verified pack
→ startup/realtime warm-up
→ offline request for the same content and exact lane
→ inspect owner receipts
→ attempt another model activation under controlled pressure
→ verify rollback preserves the old active model
```

## Review questions

An independent review should try to falsify this proposal by answering:

1. Can any live ASR, auxiliary, request-scoped materializer, split/unified pool,
   thread-local loaded context, FireRed Stream-VAD owner, or serve-batch retained
   runtime bypass the inventory facet, owner protocol, or candidate transaction?
2. Can every physical owner use serialized actor or bounded exclusive checkout,
   with serve-batch correctly represented as owner-internal scheduling plus
   resident/session footprint rather than a third policy?
3. Do `ResidentKey` and `OwnerInstanceKey` preserve source generation,
   adapter/LoRA fingerprint, representation partition, lane, decoder capacity,
   service scope, checkout slot, and instance generation without encoding thread
   identity?
4. Can any receipt classify file-backed policy residency, observed pressure,
   Metal UMA, Vulkan heap, copied/imported/backend-private, or quarantined bytes
   incorrectly?
5. Is there a state transition where V2 requested selection and active runtime
   cannot recover after termination between atomic record replacement and pointer
   exchange?
6. Do Windows replacement, staging cleanup, record validation, and startup
   reactivation tests prove the recovery protocol rather than merely invoking an
   atomic-file helper?
7. Can advisory forecast accidentally become an authorization path, especially
   when observation changes between forecast and activation?
8. Does any provider byte/domain/chunk fact leak into the family facet, or any
   model semantic/concurrency fact leak into the broker/backend adapter?
9. Can migration temporarily retain two physical owner publications, an
   unplanned JIT construction path, or two persistence authorities for one
   component?
10. Are CPU, Metal, Vulkan, CUDA, and HIP observation, weight-placement, and
    quote-confidence differences preserved rather than flattened?
11. Does the proposal preserve exact/accelerated-only fail-closed semantics,
    forbid cross-model fallback, and forbid stateful-stream replay after the
    first externally visible semantic output/decision?
12. What receipt evidence would refute the duplicate FireRed owner hypothesis,
    and does the independently confirmed persistence-first activation defect
    still justify the transaction if duplication is refuted?

## Acceptance criteria

The design is implemented only when all of the following are true:

1. every resident allocation, including request-scoped and non-ASR auxiliary
   owners, can be attributed to one canonical owner-instance receipt;
2. every live family, auxiliary, split/unified pool, and serve-batch retained
   runtime declares bounded ownership topology in the inventory projection;
3. all compiled and enumerated providers enter the same owner and activation
   contracts while preserving host/device partitions, weight-placement rules,
   and native memory semantics;
4. family facets contain no provider bytes, while the ggml/backend adapter alone
   expands physical allocation specifications;
5. production code has no unplanned family-local JIT owner, retry, publication,
   or fallback path after the Phase 3 gate;
6. model activation is admitted, materialized, attested, reconciled, persisted,
   and published as one recoverable transaction that deepens existing modules;
7. `default_selection` V2 is the only durable authority, preserves
   `Installed | NotInstalled | Unset`, and cannot be bypassed by pull/install or
   CLI fallback;
8. any activation failure or forecast/activation race preserves the previous
   durable selection and active runtime;
9. desktop neither decides physical capacity nor persists model selection;
10. startup warm-up, stateful streaming, offline execution, serve-batch, and
    split/unified routes cannot create unpriced resident multiplicity or replay
    after the semantic-output frontier;
11. owner receipts can confirm or refute the original 11.49-GB hypothesis;
12. deletion/source tests prevent old owner, retry, JIT, rebind, and persistence
    paths from returning; and
13. the real-host CPU, Metal, Vulkan, CUDA, and HIP matrix passes cold, warm,
    pressure, rollback, atomic-recovery, and receipt-attribution gates before
    release.
