# Acknowledgments

OpenASR is built on a mountain of open work. This page is our thank-you to the
projects, models, and communities that make it possible. The formal,
legally-required attributions live in [NOTICE](NOTICE) — this is the human version.

## The computational core: ggml

OpenASR's entire native runtime runs on **ggml**, the tensor library behind
**llama.cpp** and **whisper.cpp**. Every model we support executes through a thin
Rust layer over a ggml fork. None of this would exist without Georgi Gerganov and
the ggml / llama.cpp / whisper.cpp communities — their work is the foundation we
stand on, and we send fixes back upstream where we can.

- ggml — <https://github.com/ggml-org/ggml>
- llama.cpp — <https://github.com/ggml-org/llama.cpp>
- whisper.cpp — <https://github.com/ggml-org/whisper.cpp>

## The models we run

OpenASR does not train models. We re-implement open model architectures on ggml
and republish redistributable `.oasr` packs. Each pack preserves its original
authors, upstream source, revision, license, and credits — both embedded in the
pack metadata and on its page in our Hugging Face catalog:
**<https://huggingface.co/OpenASR>**

Rather than restate that information here, follow the links — every model page
credits the people who built the original.

**Speech recognition**

- Whisper — <https://huggingface.co/OpenASR/whisper-small>
- Cohere Transcribe — <https://huggingface.co/OpenASR/cohere-transcribe-03-2026>
- Qwen3-ASR — <https://huggingface.co/OpenASR/qwen3-asr-0.6b>
- Moonshine — <https://huggingface.co/OpenASR/moonshine-tiny>
- X-ASR (Zipformer) — <https://huggingface.co/OpenASR/xasr-zh-en>
- Dolphin CN-Dialect Small (DataoceanAI) — <https://huggingface.co/OpenASR/dolphin-cn-dialect-small>
- SenseVoice (FunAudioLLM, Alibaba Group; FunASR Model License v1.1) --
  <https://huggingface.co/FunAudioLLM/SenseVoiceSmall> (runs from user-imported
  packs until an OpenASR pack is published)
- Parakeet-CTC (NVIDIA NeMo) and wav2vec2 / data2vec (Meta AI) run from
  user-imported packs.
- Parakeet TDT 0.6B v3 (NVIDIA, CC-BY-4.0) --
  <https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3> (25 European languages;
  FastConformer + Token-and-Duration Transducer, trained with NeMo on the
  Granary corpus).

**Speaker diarization**

- pyannote segmentation — <https://huggingface.co/OpenASR/pyannote-segmentation-3.0>
- WeSpeaker speaker embedder — <https://huggingface.co/OpenASR/wespeaker-voxceleb-resnet34-lm>
- Silero VAD (voice activity detection) and the BUT Speech@FIT PLDA parameters
  (via the pyannote community bundle) power VAD and diarization refinement — see
  [NOTICE](NOTICE) for the vendored-asset attributions.

**Translation (experimental)**

- Hy-MT2 — <https://huggingface.co/OpenASR/hymt2-1.8b>

## Design and implementation references

OpenASR's native runtime is written from scratch, but several components are
clean-room reimplementations whose designs we learned by studying open reference
code. We did not reuse their source; the ideas and the debugging trails deserve
credit all the same:

- **icefall / k2 (Next-gen Kaldi)** — X-ASR's Zipformer2 transducer encoder,
  joiner, and streaming decoder reimplement the icefall recipe, and our importer
  follows its checkpoint tensor naming. <https://github.com/k2-fsa/icefall>
- **WeNet** — the Dolphin family reproduces WeNet's E-Branchformer encoder
  layout, state-dict naming, and attention-rescoring decode.
  <https://github.com/wenet-e2e/wenet>
- **pyannote.audio** — our pure-Rust segmentation and speaker-embedding forward
  passes port pyannote.audio's processing pipeline (beyond the model weights
  credited above). <https://github.com/pyannote/pyannote-audio>
- **sherpa-onnx (k2-fsa)** — the diarization clustering default follows
  sherpa-onnx's average-linkage approach.
  <https://github.com/k2-fsa/sherpa-onnx>
- **torchaudio** — `torchaudio.compliance.kaldi` is the numeric parity oracle
  behind our from-scratch fbank frontends. <https://github.com/pytorch/audio>
- **CrispASR** — we studied its Qwen ASR GGUF implementation while designing our
  own Qwen family runtime. <https://github.com/CrispStrobe/CrispASR>
- **Handy** — side-by-side comparison with Handy's push-to-talk dictation shaped
  our desktop insertion and recording-stop behavior.
  <https://github.com/cjpais/Handy>

## Inspiration

- **Ollama** — the "pull a model and it just runs" experience is what OpenASR
  aims to bring to speech recognition. <https://github.com/ollama/ollama>

## Model hosting

- **Hugging Face** hosts our model catalog and every `.oasr` pack we publish —
  <https://huggingface.co/OpenASR>
- **hf-mirror.com** keeps model downloads fast and reliable for users far from
  the Hub.

## Data

- The demo and test clip `fixtures/jfk.wav` is a public-domain excerpt of John F.
  Kennedy's 1961 inaugural address, distributed via the whisper.cpp project.
- The performance harness uses a LibriSpeech test-clean clip.

## The Rust ecosystem

OpenASR is written in Rust and leans on the wider crate ecosystem for audio I/O,
linear algebra, FFT, serialization, HTTP, and the CLI. The full dependency set and
its licenses live in the workspace `Cargo.toml` files and are gated by
`cargo deny`.

## And you

If you are reading this — trying OpenASR, filing an issue, or sending a patch —
thank you. See [CONTRIBUTING.md](CONTRIBUTING.md) to get involved.
