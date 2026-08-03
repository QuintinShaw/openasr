# Graph cancellation contract

OpenASR has one cooperative cancellation contract for single-job ggml graph
execution. It is implemented in the shared backend layer and used by direct and
scheduler-backed runners; model families must not add backend-specific wiring.

## Modes

`ggml_backend_graph_compute_with_abort` and its scheduler counterpart are
synchronous, compute-scoped calls. They report the mode used through an output
parameter:

- `NATIVE`: the backend exposes `ggml_backend_set_abort_callback` and reports
  the observation boundary actually used by this compute. CPU polls between
  graph nodes; Metal gates dispatches; CUDA/HIP and Vulkan stop later
  submissions. A warmed CUDA/HIP graph replay is still one indivisible launch,
  so it truthfully reports graph-completion observation.
- `SEGMENTED`: the backend has no native hook. The shared layer submits graph
  views of at most 32 nodes, synchronizes after every view, and polls before the
  first and after every completed view.
- `DISABLED`: no callback was supplied. OpenASR does not call the cancellation
  entry point in this case; it uses the original graph-compute API unchanged.

The contract is unified; the mechanism is deliberately not. Forcing a backend
with a native hook through `SEGMENTED` would replace its normal submission path
with coarser graph views and unnecessary synchronization overhead. Backends
without an equivalent hook use the shared fallback.

The 32-node segment is an explicit latency/throughput compromise. It is half of
Metal's historical 64-node main submission: reducing it lowers worst-case
observation latency but adds command submission and synchronization overhead;
increasing it improves throughput but delays cancellation. This is a node-count
bound, not a time bound. A single long-running GPU kernel cannot be preempted.

The source-build matrix is:

| Backend | Mode | Cancellation checkpoint |
| --- | --- | --- |
| CPU | `NATIVE` | backend per-node poll |
| Metal | `NATIVE` | host-visible abort flag sampled before dispatch |
| CUDA / HIP | `NATIVE` | submission checkpoints; warmed replay at graph completion |
| Vulkan | `NATIVE` | per-context submission/timeline completion |
| SYCL | `SEGMENTED` | synchronized graph views |

Future backends automatically receive `SEGMENTED` behavior unless they expose
the native registry proc. There is no supported silent no-op mode.

## Lifetime and scheduler rules

The current job owns an `Arc<AtomicBool>`. Rust clones it immediately before
compute and passes its pointer only to the synchronous FFI call. The shared
layer retains no callback or job data after return. This makes cached backends
safe across sequential jobs and keeps parallel worker threads isolated.

The scheduler polls before graph allocation, before and after every
scheduler-controlled input transfer/wait boundary, before each split compute,
and inside each backend compute. Its reported mode is `SEGMENTED` if any
scheduler backend lacks a native hook; otherwise it is `NATIVE`. It also reports
the coarsest observation boundary across the participating backends. A backend
API call already entered by the scheduler (one graph replay, event wait,
synchronization, copy, or kernel) is indivisible, so cancellation cannot preempt
that call; it is observed at the next reported checkpoint. An aborted return is
synchronized: no submitted transfer or graph work remains in flight and can
touch buffers reused by a later job.

Cancel unwinds the active request; it does not evict stateless cached backend or
device handles. Model/session tensor state is different: native and segmented
compute can stop after a prefix of side-effect nodes has written resident KV or
other persistent tensors. Any non-successful compute therefore poisons its graph
session. The same graph rejects later uploads/computes until its owner drops and
rebuilds it. Shared `Seq2SeqReusableDecodeGraph` and `LlmReusableDecodeGraph`
owners include poison in their reuse-match decision. The Qwen whole-decoder
abstraction also propagates failures from temporary prefill graphs that borrow
its resident KV arena, and its qwen/firered-llm/moss-td cache handoff discards a
poisoned graph while retaining uploaded weights and the backend handle.

Whisper cross-cache population and Cohere/Moonshine cross-cache population use
request setup graphs: a failure exits before decode, and the next request
rewrites every cache layer before reading it. Their incremental persistent graph
is still poisoned and rebuilt. X-ASR's persistent encoder graph likewise treats
a poisoned session as a reuse miss. Cache eviction of otherwise healthy models
remains owned by the explicit idle-unload path. Pause is different from cancel:
it waits only at long-form slice boundaries and never arms the graph callback.

Serve-batch graphs contain work for multiple jobs. A single member's cancel
flag must not abort the shared graph and harm healthy siblings. Serve-batch
paths that support per-request cancellation therefore carry each job's control
explicitly and remove canceled members at typed prefill-chunk/token boundaries
(currently Qwen); single-job graph execution uses L2.

Before any long-form graph exists, the shared slice planner carries that same
request control into VAD. Built-in FireRed Stream-VAD and energy VAD providers
poll between bounded recording chunks and return a typed planning cancel;
packed-audio admission and materialization are skipped after cancellation. The
legacy provider method remains available for callers with no request control,
while the cancellable trait method gives external providers an override point
without putting family-specific cancellation code in the native executor.

## Boundary

OpenASR invokes `ggml_backend_graph_compute` and scheduler graph compute; it
does not invoke ggml's opaque backend graph-plan compute API. The latter cannot
be segmented after plan creation and is outside the cancellation contract
described above. OpenASR helpers named “graph plan” build ordinary
`ggml_cgraph` values and are covered.

## Regression evidence

The ggml fake-backend contract test proves the shared segmented fallback's
callback-free single submission, `false` completion as `32 + 32 + 1`, a
deterministic mid-graph flip after the first synchronized segment, typed abort,
and no pending work on return. Native backend tests separately cover their
reported observation boundaries. A
two-backend scheduler seam also forces an incompatible-buffer copy, flips cancel
inside that copy, proves the destination split is never submitted, and proves
the pending copy is drained before return. Rust tests cover callback false/true,
direct and scheduler paths, parallel-job isolation, real Metal native
cancellation, Metal scheduler cancellation, callback-false completion,
cached-Metal reuse after a canceled job, and persistent-session poison/rebuild
including a synthetic cancellation-contract error. Long-form tests additionally
prove a cancellation
raised inside a provider remains typed through planning, skips packed-audio
materialization, and maps to `BackendError::TranscriptionCanceled`.
