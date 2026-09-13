# Roadmap

Development priorities, not a second feature inventory. Current behavior is
defined by code and tests; [Architecture](../ARCHITECTURE.md) maps the code,
[Known Limitations](KNOWN_LIMITATIONS.md) records user-visible gaps, and
[model cards](../model-registry/models/) describe individual packs.

## Active priorities

### Architecture cohesion

- Keep public APIs small and ownership explicit across core, server, and clients.
- Remove obsolete compatibility paths and duplicated state.
- Keep family-specific behavior behind the shared model lifecycle and executor
  contracts; new families must reuse the shared admission and validation gates.

### Correctness and qualification

- Close gaps in file-task control and timeline diagnostics without weakening
  cancellation, authentication, or fail-closed behavior.
- Broaden long-form, multilingual, timestamp, and real-device validation beyond
  short smoke tests. Pack publication alone is not hardware qualification.
- Keep remote-compute acceptance tied to client/server end-to-end evidence.

### Performance

- Keep portable GGUF-backed `.oasr` packs canonical, without embedded
  platform-specific compute caches.
- Improve performance against the committed regression gates. Claims must name
  the measured pack, device, workload, and baseline.

## Boundaries

- Homebrew, binary archives, and Docker distribution already exist; see
  [installation](../README.md#for-developers) and [Releasing](../RELEASING.md).
- Broad streaming-quality and multilingual guarantees remain qualification work,
  not promises inferred from a family being registered or a pack being public.
- Model onboarding follows [Model Onboarding](MODEL_ONBOARDING.md), rather than
  adding parallel runtime or packaging paths.

## Validation

Run targeted checks first, then the applicable regression gates. Record evidence
with the change; retain durable contracts here, not completed implementation
checklists or session-by-session test diaries.
