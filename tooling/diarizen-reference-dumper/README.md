# DiariZen Large-s80-v2 reference dumper

This development-only tool emits stage-by-stage PyTorch goldens for the native
OpenASR segmenter. It uses a deterministic synthetic waveform and never needs
meeting audio. The pinned weights are CC BY-NC 4.0; keep the checkpoint and
generated artifacts outside git.

```bash
python dump_golden.py \
  --diarizen-source /path/to/DiariZen \
  --checkpoint /path/to/pytorch_model.bin \
  --config /path/to/config.toml \
  --out-dir /path/to/tmp/diarizen-golden
```

The native parity test consumes `diarizen_large_s80_v2_golden.npz` through a local
environment variable and remains ignored in ordinary CI.
