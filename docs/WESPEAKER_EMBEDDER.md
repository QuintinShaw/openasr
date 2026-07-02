# WeSpeaker ResNet34 speaker embedder

OpenASR supports pyannote/WeSpeaker ResNet34 as its speaker embedder.
The default runtime path resolves it with:

```bash
OPENASR_WESPEAKER_PACK=/path/to/wespeaker-voxceleb-resnet34-lm-f32.oasr
```

`OPENASR_WESPEAKER_PACK` is optional when the pack is installed under
`openasr_home()/models/wespeaker-voxceleb-resnet34-lm/<quant>/`.

## Source and license

The implemented weight source is:

- Hugging Face model: `pyannote/wespeaker-voxceleb-resnet34-LM`
- Revision used for this port: `837717ddb9ff5507820346191109dc79c958d614`
- Architecture: pyannote.audio 3.1.1 `WeSpeakerResNet34`, ResNet34 `[3,4,6,3]`, 256-d output, about 6.6M parameters
- Weight license: `CC-BY-4.0`

Do not mark these weights as Apache-2.0. The local `.oasr` importer records
`openasr.license.name=CC-BY-4.0`, `openasr.source.name`, and
`openasr.source.revision` in pack metadata.

## Frontend

The WeSpeaker/pyannote frontend uses 16 kHz mono audio, 80-bin Kaldi log-mel
fbank, 25 ms windows, 10 ms shift, `dither=0`, DC-offset removal, pre-emphasis
`0.97`, power spectrum, log fbank, Hamming windowing, pyannote's `waveform *
32768` scaling, and per-utterance CMN.

## Local pack build

The reference-safetensors + `import wespeaker` recipe lives in
[Diarization Pack Publishing](DIARIZATION_PACK_PUBLISHING.md); the signed catalog
publishes the result as the `wespeaker-voxceleb-resnet34-lm:f32` capability pack.

The provenance contract is load-bearing: the generated safetensors must carry
`__metadata__.source_name`, `__metadata__.source_revision`, and
`__metadata__.license` for the pinned source above, and the local importer fails
closed when those fields are missing or mismatched. The WSR1 parity golden stores
the source name, source revision, and checkpoint SHA-256 before its per-case
records so parity failures identify stale or misattributed oracle files clearly.

## Validation snapshot

Local reference artifacts generated from the pinned HF checkpoint:

- fbank parity vs torchaudio: max abs error `0.000507` across three synthetic
  clips plus `fixtures/jfk.wav`
- network parity vs Python reference: cosine `>= 0.99999976`
- end-to-end embedder parity: cosine `>= 1.00000000` within float precision
- `.oasr` round-trip: byte-identical network output vs safetensors on synthetic
  features
- release-mode embedding RTF on `fixtures/jfk.wav` with
  `RAYON_NUM_THREADS=8`:
  - WeSpeaker ResNet34 f32: `0.03460`

Realtime speaker-change detection uses the same WeSpeaker embedder as
identity/label attribution. At f32 RTF `0.03460`, a 2.5 s detector window once
per second projects to about `0.087` added RTF.
