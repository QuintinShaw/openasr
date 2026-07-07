# Test and demo audio fixture

## `jfk.wav`

The single audio fixture used across the project: the README quick start, CLI
demos, the default benchmark clip, and the test suite.

- Provenance: `samples/jfk.wav` from the ggml-org **llama.cpp / whisper.cpp**
  project (MIT-licensed repository). The recording is a public-domain excerpt of
  John F. Kennedy's 1961 inaugural address.
- Format: 16 kHz mono 16-bit PCM WAV, 11.0 seconds.
- SHA-256: `59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e`.

See [`NOTICE`](../NOTICE) for attribution.

## `zh_sample.wav`

A Mandarin Chinese test fixture used for firered-aed golden-diff parity tests.

- Provenance: synthesized locally with macOS `say -v Tingting` from an
  original sentence written for this test (no third-party recording, no
  licensing concerns).
- Format: 16 kHz mono 16-bit PCM WAV, ~18.2 seconds.
- SHA-256: `05cfba62e07d74cb7a90c3ba6f6aa185653d8da387c2f5df5f82172ec073340e`.
