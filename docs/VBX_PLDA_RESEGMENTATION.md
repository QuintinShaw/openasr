# PLDA mixture and HMM VBx resegmentation

OpenASR includes offline-only PLDA refinement paths for file diarization. They
run after AHC initialization and before final speaker-turn reconstruction.
Realtime diarization remains on the existing streaming path.

## Source and license

The PLDA/LDA numeric parameters come from the public Hugging Face mirror:

- Source: `pyannote-community/speaker-diarization-community-1`
- Revision: `8a527374977391da736e0daaef26855d949d9685`
- Files: `plda/plda.npz`, `plda/xvec_transform.npz`
- License: `CC-BY-4.0`
- Upstream acknowledgement: the PLDA model is credited by pyannote to the BUT
  Speech@FIT group.

OpenASR converts those arrays into a little-endian f32 asset
(`diarize/vbx/assets/community1_plda_f32.bin`) and applies the same transform
chain as pyannote's `vbx_setup` / `utils/vbx.py` over OpenASR-owned arrays. The
default dense resegmentation is a PLDA-mixture responsibility update; an HMM VBx
variant (log-domain forward-backward) is available behind
`OPENASR_DIAR_DENSE_VBX_VARIANT=hmm`. No BUT VBx implementation code is vendored
or copied.

## Runtime scope

The pass is intentionally conservative:

- only 256-dimensional WeSpeaker embeddings are eligible;
- only context-rich, dense files with clear AHC oversegmentation enter the pass;
- cannot-link constraints from simultaneous pyannote regions are preserved;
- short files, the real-time path, and CAM++ fallback diarization keep the
  previous AHC behavior.

The default pass enables PLDA-constrained state merging and PLDA mixture
resegmentation over 1.5s windows with a 0.25s shift across the speech mask. Long
segmentation regions are chunked for embedding so dense responsibilities can
split speaker changes inside an otherwise under-split local-speaker span. Final
turn construction preserves pyannote local-speaker overlap constraints, maps
short regions through nearest compatible embedded chunks, and collapses adjacent
same-speaker/same-overlap turns after either reconstruction branch.

Current dense resegmentation defaults:

- `Fa = 0.1`
- `Fb = 5.0`
- initialization smoothing `= 7.0`
- max iterations `= 20`
- HMM loop probability `= 0.95` when the HMM variant is selected

Debug overrides:

- `OPENASR_DIAR_VBX=0` disables PLDA and dense resegmentation refinement.
- `OPENASR_DIAR_DENSE_VBX=0` disables only dense resegmentation.
- `OPENASR_DIAR_DENSE_VBX_VARIANT=plda_mixture` selects the default PLDA
  mixture resegmentation.
- `OPENASR_DIAR_DENSE_VBX_VARIANT=hmm` selects the HMM VBx variant.
- `OPENASR_DIAR_DENSE_VBX_HMM=1` is a shorthand for the HMM variant.
- `OPENASR_DIAR_VBX_MERGE_THRESHOLD` overrides the PLDA merge threshold.
- `OPENASR_DIAR_DENSE_VBX_FA`, `OPENASR_DIAR_DENSE_VBX_FB`,
  `OPENASR_DIAR_DENSE_VBX_INIT_SMOOTHING`, and
  `OPENASR_DIAR_DENSE_VBX_MAX_ITERS` override dense resegmentation parameters.
- `OPENASR_DIAR_DENSE_VBX_LOOP_PROB` overrides the HMM transition loop
  probability.
- `OPENASR_DIAR_DENSE_VBX_CACHE` is a developer-only cache for transformed
  dense windows during parameter sweeps. Cache reads require strict metadata
  equality: audio hash, sample rate, embedder fingerprint/dimension, window
  constants, and PLDA asset id must all match.

The current `Fa`/`Fb`/smoothing/chunk constants are small-set-tuned defaults,
not per-file tuning knobs; do not change them between files when comparing
Ali/Vox DER.
