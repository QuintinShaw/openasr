# Known Limitations

This page lists current user-visible limits. For implementation truth and
sequencing, see [Roadmap](ROADMAP.md) (Implemented-baseline section).

## Current limitations

- OpenASR is Apache-2.0 open core, but there is no public binary release yet: no
  published binaries/installers/checksums, and no package-manager channels. Build
  from source meanwhile. Public model-pack distribution is limited to catalog
  entries explicitly marked `public:true`.
- The only executable backends are the default `native` and the opt-in `mock`
  stub. Native transcription runs offline from `.oasr` runtime packs -- a pinned
  `--model-pack`, an installed pack, or one the CLI installs on first use through
  a visible consent prompt -- and stays fail-closed by stage.
- The consent-gated CLI pull, the no-silent-download boundary, and pull/install
  mechanics are centralized in [Model Catalog, Registry, and Distribution](MODEL_CATALOG_ARCHITECTURE.md);
  the HTTP server never pulls.
- Realtime cadence is descriptor-driven: any pack whose family has a registered
  streaming executor gets live partials, and every built-in ASR family
  (Qwen3-ASR, Whisper, Cohere Transcribe, Moonshine, Parakeet-CTC, wav2vec2-CTC,
  SenseVoice, Dolphin, and X-ASR) registers one -- a startup completeness gate
  rejects any family that does not. X-ASR is frame-sync (append-only partials);
  every other family re-decodes a growing window (may revise partials, FINAL is
  byte-identical to offline). Official published packs with public product
  guarantees are still pending.
- Speaker diarization is opt-in (`--diarize` / the API `diarize` flag). It uses
  the ReDimNet2-B6 speaker-embedding pack and optional pyannote segmentation
  capability packs (pulled/installed on demand) to attribute anonymous `SPEAKER_NN`
  labels onto any model's transcript; without ReDimNet2-B6 installed, diarize fails
  closed. Cohere packs that declare
  `openasr.features.diarization=cohere-token-stream-v1` can additionally emit
  inline speaker tokens. Without the required capability pack a diarize request
  fails closed rather than fabricating speaker labels. Labels are session-relative
  (`SPEAKER_00/01`, ...) unless the speaker has been enrolled through the opt-in,
  operator-only Voice ID system (`/v1/voice-id/*`; see [FAQ.md](FAQ.md#is-diarization-available)
  for enrollment/naming requirements); an unenrolled speaker, or one whose speech
  falls short of Voice ID's naming evidence gate, still only gets a
  session-relative number. See [SECURITY.md](../SECURITY.md) for the diarization
  privacy model and the remote-mode trust contract.
- Phrase bias / hotword boosting is implemented for the native runtime decode
  path. Requests still fail closed when the selected model tokenizer cannot
  encode a requested phrase, and the mock backend still rejects non-empty
  phrase-bias requests. Most families apply phrase bias as a per-token logit
  boost during decoding; Dolphin instead runs its upstream-trained native
  deep-biasing context module (`context_module.*`: a BiLSTM context encoder +
  cross-attention fusion into the encoder output ahead of attention rescoring,
  arXiv:2305.12493) and therefore has no per-phrase `boost` weight to apply --
  the phrase *list* is the only signal it honors, and CTC prefix-beam n-best
  generation still runs over the un-biased encoder output (only the
  attention-rescoring decoder input is biased), matching the upstream
  reference decode.
- Word-level timestamp requests are accepted and exported in JSON/VTT. Whisper
  uses native decoder cross-attention frame probabilities and the CTC families
  (parakeet-ctc, wav2vec2-ctc) use decoder frame spans; Qwen, Cohere, and
  Moonshine fall back to decoder token-position estimates because those runtimes
  do not expose acoustic attention, so their word timings are approximate.
  Dolphin does not emit word-level timestamps at all -- its CTC/attention joint
  decode only returns a single segment-level span, so `--word-timestamps`
  requests against a Dolphin pack yield an empty word list rather than an error.
  SenseVoice likewise returns an empty word list: its CTC frames sit on a 60 ms
  low-frame-rate grid behind 4 prompt frames, so per-word times would be
  fabricated precision rather than acoustic timestamps. FireRedASR-AED also
  returns an empty word list -- its attention-based encoder-decoder has no
  alignment head exposed yet, so `--word-timestamps` requests yield a
  segment-level span rather than fabricated per-word times.
- Word-timestamp refinement (`--word-timestamps=aligned` / API
  `timestamp_granularities=word_aligned`) is an opt-in tier on top of the
  approximate timestamps above: it re-runs the finished transcript and the full
  source audio through the Qwen3-ForcedAligner-0.6B capability pack and
  replaces each segment's word spans with the aligner's own output. Passing
  `=aligned` is the consent to install the pack (mirroring `--diarize`'s
  ReDimNet2-B6 auto-install); the native backend never installs it silently
  otherwise, and the server never auto-installs it at all (operator-gated pull
  only). It is native-backend-only, requires `--word-timestamps` semantics
  implicitly (word timestamps are always emitted for `aligned`), and does not
  yet support Japanese or Korean transcripts -- the reference routes those
  through external morphological segmenters (`nagisa`/`soynlp`) that have not
  been ported, so an `aligned` request against ja/ko text fails closed with a
  typed error rather than mis-tokenizing. Every other family keeps its
  approximate timestamps unchanged; `aligned` only affects the words replaced
  by the aligner's output, never segment text or speaker attribution.
- Hardware execution target selection is generic: Desktop/server requests support
  `auto`, `cpu`, and `accelerated` when the native runtime reports an accelerated
  device. There is no public per-provider/per-device pinning surface such as
  `gpu0`. Internally the runtime can resolve a concrete execution route
  (`provider` + ggml stable device name + optional PCI `device_id` from CUDA/HIP,
  and from Vulkan when available). What is route-isolated today:
  - thread-local ggml **backend-handle** cache (Exact pin never shares a handle;
    preferred/Auto may Optimus-fall through discrete -> iGPU but always caches
    under the device that actually initialized)
  - streaming **worker** keys
  What is intentionally **not** route-isolated yet (hard gate before public Exact):
  - family prepared-runtime caches remain coarse `(path, backend)`
  - serve-batch engine keys do not include route
  - **admission capacity stays per model identity** (CPU and accelerated share one
    slot for the same model; route does not multiply capacity)
  Exact device pins are fail-closed: missing devices, init failures, Metal
  (still `MTLCreateSystemDefaultDevice` only), and CPU StableId Exact return typed
  not-found / not-addressable / init-failed errors instead of silently swapping
  cards or falling back to CPU. Unavailable coarse `accelerated` targets still
  fail closed. Physical PCI keys are normalized (trim + lower-case) only; full
  BDF grammar validation is a follow-up.
- No public reproducible real-backend benchmark or long-audio stability evidence
  is published. The performance harness, regression gates, and competitive
  comparisons are internal (see [Performance](../perf/PERFORMANCE.md)); no claim of
  having finally beaten open-source baselines is made — only that the harness and
  gates are in place.
- No public quality/WER guarantee is claimed for longform timestamps/exports;
  these are validated on internal smoke lanes only.
- Cohere longform carries a model-specific safety policy on top of the shared
  planner contract (chunk cap, no overlap, prompt carry disabled, Metal multichunk
  prefers CPU decoder). It matches current correctness/perf evidence and is not yet
  generalized into a model-agnostic runtime tuning layer.
- System-audio capture now has native smoke backends on macOS, Windows, and
  Linux. Windows additionally supports per-process loopback capture
  (`run_process_loopback_capture`, via `AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK`)
  alongside its existing all-system WASAPI loopback path; this requires
  Windows 10 2004 (build 19041) or later and is capability-gated by
  `process_loopback_support()`, which performs a real runtime activation
  probe (not an OS-version heuristic, so it cannot be fooled by application
  compatibility shims) and fails closed to "unsupported" on older systems.
  `list_candidate_processes()` enumerates pid + executable name via a
  Toolhelp32 snapshot for callers building a process picker. macOS and Linux
  do not implement per-process capture yet -- both report it as `unsupported`
  through the same capability probe rather than emulating it or silently
  falling back to all-system capture; macOS keeps its Core Audio process-tap
  all-system path and Linux keeps its `pactl`/`parec` monitor-source capture.
  Windows real playback smoke (both all-system and per-process) has been
  executed on a real Windows 11 session; Linux real playback smoke still
  needs to be executed on a Linux session. Per-process capture has no desktop
  UI wiring yet -- it is a library-level API only.
- `serve` is single-model: it runs the one pack resolved at launch
  (`--model-pack` / an installed `--model`). There is no per-request lazy model
  loading or an `openasr ps`-style multi-model runner yet -- restart `serve` to
  switch models.
- `openasr pull` always fetches and re-installs; it has no incremental update
  (no revision/digest diff, `up to date` check, or `--force`). Re-pulling a model
  re-downloads it.
- Source-language control is per-model and capability-gated (see
  `openasr show <pack>` / `/v1/capabilities`). Multilingual Whisper auto-detects an
  unset language and accepts an explicit `--language`; Cohere and the English-only
  families resolve to their fixed/default language. Qwen3-ASR auto-detects
  internally but does **not** expose the detected language, so its reported
  `language` is null and an explicit `--language` is rejected rather than silently
  ignored -- use a multilingual Whisper pack when you need to force or read back the
  language. (Wiring Qwen's text-prompt language conditioning is tracked, but needs a
  real-pack parity check against the reference inference before it can be claimed.)
  Dolphin is specify-only: it does not auto-detect, so an explicit `--language`
  selects one of its 14 recognition codes (`zh` plus 13 Chinese regional-dialect
  codes such as `zh-sichuan`, `zh-shanghai`, `zh-hebei`) via a decode-prompt
  token, defaulting to `zh` when unset; an unsupported code is rejected rather
  than silently falling back.
  SenseVoice accepts an explicit `zh`/`yue`/`en`/`ja`/`ko` selection via its
  4-token decode prompt and auto-detects when unset (the model emits a readable
  language tag); an unsupported code is rejected fail-closed, and a detected
  code outside the advertised set reports `language: null` rather than a guess.
  SenseVoice also classifies emotion and audio events internally, but those
  tags are intentionally not exposed on the API surface yet.
- Cooperative cancel of an in-flight offline transcription is layered: long-form
  slice boundaries (L0), shared seq2seq greedy token / prefill chunk checks (L1),
  and a compute-scoped ggml abort contract (L2). CPU uses its native per-node
  callback. Metal and source-enabled CUDA/HIP/Vulkan/SYCL builds use the shared
  backend fallback: while cancel is armed, a graph is submitted in at most
  32-node views, synchronized, then polled. A scheduler reports the weakest mode
  among all of its backends and applies the same contract to every split, with
  checks around scheduler input transfers as well, so a missing backend-specific
  proc is never a silent no-op. One already-entered kernel, event wait, or copy
  remains non-preemptible; the bound is a graph/scheduler checkpoint, not a
  wall-clock deadline. A failed persistent compute poisons its model/session
  graph and forces graph/KV rebuild before reuse; cached backend handles and
  uploaded immutable weights are retained. See
  [Graph cancellation contract](design/graph-cancellation.md).
  Pause still only blocks at slice boundaries and never arms graph cancellation.

## What works now

See [Roadmap](ROADMAP.md) (Implemented-baseline section) for the current
working behavior matrix.

## Related docs

- [Model Catalog, Registry, and Distribution](MODEL_CATALOG_ARCHITECTURE.md)
- [Roadmap](ROADMAP.md)
- [FAQ](FAQ.md)
- [Docs Index](DOCS_INDEX.md)
