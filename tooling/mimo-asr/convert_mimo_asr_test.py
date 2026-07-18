#!/usr/bin/env python3
"""Conversion-correctness tests for the MiMo-V2.5-ASR -> .oasr converter.

Covers the pure remap/type/metadata logic plus a full tiny synthetic
safetensors -> GGUF round-trip (no real weights needed). The three P2.0
blood-lesson corrections have explicit assertions:

  * skip@L3  -- ``mimo.tok.encoder.skip_layer_id == 3`` and both the captured
    layer (skip_id-1) and the final encoder layer are preserved.
  * conv1 stride 1 / conv2 stride 2.
  * 8-codebook *summation* semantics + exactly 8 speech_embd tables with the
    per-channel vocab sizes.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import convert_mimo_asr as C


# ---------------------------------------------------------------------------
# pure remap
# ---------------------------------------------------------------------------

class RemapMainTest(unittest.TestCase):
    def test_top_level(self):
        self.assertEqual(C.remap_main_tensor("model.embed_tokens.weight"), "token_embd.weight")
        self.assertEqual(C.remap_main_tensor("model.norm.weight"), "output_norm.weight")
        self.assertEqual(C.remap_main_tensor("lm_head.weight"), "output.weight")
        self.assertEqual(C.remap_main_tensor("speech_group_downcast.weight"), "speech_group_proj.weight")

    def test_backbone_layer(self):
        self.assertEqual(C.remap_main_tensor("model.layers.5.self_attn.q_proj.weight"), "blk.5.attn_q.weight")
        self.assertEqual(C.remap_main_tensor("model.layers.5.self_attn.q_proj.bias"), "blk.5.attn_q.bias")
        self.assertEqual(C.remap_main_tensor("model.layers.5.self_attn.k_proj.bias"), "blk.5.attn_k.bias")
        self.assertEqual(C.remap_main_tensor("model.layers.5.self_attn.o_proj.weight"), "blk.5.attn_output.weight")
        self.assertEqual(C.remap_main_tensor("model.layers.5.mlp.gate_proj.weight"), "blk.5.ffn_gate.weight")
        self.assertEqual(C.remap_main_tensor("model.layers.5.post_attention_layernorm.weight"), "blk.5.ffn_norm.weight")

    def test_input_local_and_speech(self):
        self.assertEqual(C.remap_main_tensor("input_local_transformer.layers.3.self_attn.v_proj.bias"), "inlocal.blk.3.attn_v.bias")
        self.assertEqual(C.remap_main_tensor("input_local_transformer.norm.weight"), "inlocal.norm.weight")
        self.assertEqual(C.remap_main_tensor("speech_embeddings.0.weight"), "speech_embd.0.weight")
        self.assertEqual(C.remap_main_tensor("speech_embeddings.7.weight"), "speech_embd.7.weight")

    def test_discards(self):
        self.assertIsNone(C.remap_main_tensor("local_transformer.layers.0.self_attn.q_proj.weight"))
        self.assertIsNone(C.remap_main_tensor("local_transformer.norm.weight"))
        self.assertIsNone(C.remap_main_tensor("local_transformer_lm_heads.0.weight"))
        self.assertIsNone(C.remap_main_tensor("hidden_states_downcast.weight"))


class RemapTokTest(unittest.TestCase):
    def test_conv_and_norms(self):
        self.assertEqual(C.remap_tok_tensor("encoder.conv1.weight"), "audiotok.conv1.weight")
        self.assertEqual(C.remap_tok_tensor("encoder.conv2.bias"), "audiotok.conv2.bias")
        self.assertEqual(C.remap_tok_tensor("encoder.layer_norm.weight"), "audiotok.norm.weight")
        self.assertEqual(C.remap_tok_tensor("encoder.layer_norm.bias"), "audiotok.norm.bias")
        self.assertEqual(C.remap_tok_tensor("encoder.down_sample_layer.0.weight"), "audiotok.down_sample.weight")
        self.assertEqual(C.remap_tok_tensor("encoder.down_sample_norm.bias"), "audiotok.down_sample_norm.bias")

    def test_encoder_layer_asymmetric_qkv(self):
        self.assertEqual(C.remap_tok_tensor("encoder.layers.9.self_attn.q_proj.bias"), "audiotok.blk.9.attn_q.bias")
        self.assertEqual(C.remap_tok_tensor("encoder.layers.9.self_attn.k_proj.weight"), "audiotok.blk.9.attn_k.weight")
        self.assertEqual(C.remap_tok_tensor("encoder.layers.9.self_attn.v_proj.bias"), "audiotok.blk.9.attn_v.bias")
        self.assertEqual(C.remap_tok_tensor("encoder.layers.9.self_attn.out_proj.weight"), "audiotok.blk.9.attn_out.weight")
        self.assertEqual(C.remap_tok_tensor("encoder.layers.9.fc1.weight"), "audiotok.blk.9.ffn_up.weight")
        self.assertEqual(C.remap_tok_tensor("encoder.layers.9.fc2.bias"), "audiotok.blk.9.ffn_down.bias")
        self.assertEqual(C.remap_tok_tensor("encoder.layers.9.final_layer_norm.weight"), "audiotok.blk.9.ffn_norm.weight")

    def test_rvq_keeps_first_eight_embed_only(self):
        self.assertEqual(C.remap_tok_tensor("encoder.quantizer.vq.layers.0._codebook.embed"), "audiotok.quant.0.codebook")
        self.assertEqual(C.remap_tok_tensor("encoder.quantizer.vq.layers.7._codebook.embed"), "audiotok.quant.7.codebook")
        # level >= 8 dropped (residual causality)
        self.assertIsNone(C.remap_tok_tensor("encoder.quantizer.vq.layers.8._codebook.embed"))
        self.assertIsNone(C.remap_tok_tensor("encoder.quantizer.vq.layers.19._codebook.embed"))
        # EMA buffers dropped
        self.assertIsNone(C.remap_tok_tensor("encoder.quantizer.vq.layers.0._codebook.embed_avg"))
        self.assertIsNone(C.remap_tok_tensor("encoder.quantizer.vq.layers.0._codebook.cluster_size"))
        self.assertIsNone(C.remap_tok_tensor("encoder.quantizer.vq.layers.0._codebook.inited"))

    def test_decoder_dropped(self):
        self.assertIsNone(C.remap_tok_tensor("decoder.layers.0.fc1.weight"))
        self.assertIsNone(C.remap_tok_tensor("decoder.vocoder.something.weight"))
        self.assertIsNone(C.remap_tok_tensor("decoder.layer_norm.weight"))


# ---------------------------------------------------------------------------
# quant / dtype policy
# ---------------------------------------------------------------------------

class TypePolicyTest(unittest.TestCase):
    def test_backbone_quantized_only_when_aligned_in_q8_pack(self):
        self.assertEqual(C.choose_tensor_type("blk.0.attn_q.weight", (4096, 4096), "q8_0"), "q8_0")
        self.assertEqual(C.choose_tensor_type("token_embd.weight", (151680, 4096), "q8_0"), "q8_0")
        self.assertEqual(C.choose_tensor_type("output.weight", (151680, 4096), "q8_0"), "q8_0")
        # inner dim not 32-aligned -> falls back to f16
        self.assertEqual(C.choose_tensor_type("blk.0.attn_q.weight", (8, 30), "q8_0"), "f16")

    def test_fp16_pack_never_quantizes(self):
        self.assertEqual(C.choose_tensor_type("blk.0.attn_q.weight", (4096, 4096), "fp16"), "f16")
        self.assertEqual(C.choose_tensor_type("token_embd.weight", (151680, 4096), "fp16"), "f16")

    def test_audio_side_stays_f16_in_q8_pack(self):
        self.assertEqual(C.choose_tensor_type("audiotok.blk.0.attn_q.weight", (1280, 1280), "q8_0"), "f16")
        self.assertEqual(C.choose_tensor_type("audiotok.conv1.weight", (1280, 128, 3), "q8_0"), "f16")
        self.assertEqual(C.choose_tensor_type("inlocal.blk.0.ffn_gate.weight", (4096, 1024), "q8_0"), "f16")
        self.assertEqual(C.choose_tensor_type("speech_embd.0.weight", (1025, 1024), "q8_0"), "f16")

    def test_forced_f32(self):
        self.assertEqual(C.choose_tensor_type("blk.0.attn_q.bias", (4096,), "q8_0"), "f32")
        self.assertEqual(C.choose_tensor_type("blk.0.attn_norm.weight", (4096,), "q8_0"), "f32")
        self.assertEqual(C.choose_tensor_type("output_norm.weight", (4096,), "q8_0"), "f32")
        self.assertEqual(C.choose_tensor_type("audiotok.norm.weight", (1280,), "q8_0"), "f32")
        self.assertEqual(C.choose_tensor_type("audiotok.quant.0.codebook", (1024, 1280), "q8_0"), "f32")
        self.assertEqual(C.choose_tensor_type("audiotok.mel_filters", (481, 128), "q8_0"), "f32")


# ---------------------------------------------------------------------------
# 8-codebook summation (blood lesson #3)
# ---------------------------------------------------------------------------

class SpeechEmbeddingSumTest(unittest.TestCase):
    def test_sum_matches_manual_with_zeroemb_masking(self):
        rng = np.random.default_rng(0)
        dim = 5
        vocab = [5, 5, 3, 3, 3, 3, 3, 3]
        zeroemb = [4, 4, 2, 2, 2, 2, 2, 2]
        tables = [rng.standard_normal((v, dim)).astype(np.float32) for v in vocab]
        # T=3 frames, 8 channels; frame 1 sets channel 0 to its zeroemb id
        codes = np.array([
            [0, 1, 0, 1, 2, 0, 1, 2],
            [4, 0, 1, 2, 0, 1, 2, 0],   # channel 0 == zeroemb -> masked out
            [1, 2, 2, 0, 1, 2, 0, 1],
        ], dtype=np.int64)

        got = C.sum_speech_embeddings(tables, codes, zeroemb)

        expected = np.zeros((3, dim), dtype=np.float64)
        for t in range(3):
            for ch in range(8):
                cid = codes[t, ch]
                if cid == zeroemb[ch]:
                    continue
                expected[t] += tables[ch][cid]
        np.testing.assert_allclose(got, expected, rtol=0, atol=0)

        # controlled masking: channel 0 hits its zeroemb id; every other channel
        # uses id 0 (!= its zeroemb). The result must equal the sum of channels
        # 1..7 only, i.e. dropping channel 0's (non-zero) row.
        single = np.array([[4, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
        got1 = C.sum_speech_embeddings(tables, single, zeroemb)
        masked = sum(tables[ch][0].astype(np.float64) for ch in range(1, 8))
        np.testing.assert_allclose(got1[0], masked, rtol=0, atol=1e-6)
        # and channel 0's row was genuinely non-zero, so masking mattered
        self.assertGreater(np.abs(tables[0][4]).sum(), 0.0)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            C.sum_speech_embeddings([np.zeros((2, 2), np.float32)], np.zeros((1, 1), np.int64), [0, 0])


# ---------------------------------------------------------------------------
# metadata (blood lessons #1 + #2 encoded here)
# ---------------------------------------------------------------------------

def _meta_dict(items):
    return {it.key: it.value for it in items}


class MetadataTest(unittest.TestCase):
    def setUp(self):
        self.main, self.tok = _tiny_hparams()
        self.items = C.build_metadata(self.main, self.tok, "mimo-v2.5-asr-q8_0", "q8_0")
        self.md = _meta_dict(self.items)

    def test_skip_layer_id(self):
        self.assertEqual(self.md["mimo.tok.encoder.skip_layer_id"], 3)

    def test_conv_strides(self):
        self.assertEqual(self.md["mimo.tok.conv1.stride"], 1)
        self.assertEqual(self.md["mimo.tok.conv2.stride"], 2)

    def test_audio_channels_and_codebooks(self):
        self.assertEqual(self.md["mimo.audio.channels"], 8)
        self.assertEqual(self.md["mimo.tok.rvq.codebook_sizes"], [5, 5, 3, 3])
        self.assertEqual(self.md["mimo.speech.vocab_size"], [5, 5, 3, 3, 3, 3, 3, 3])
        self.assertEqual(self.md["mimo.speech.zeroemb_idx"], [4, 4, 2, 2, 2, 2, 2, 2])

    def test_backbone_flags(self):
        self.assertTrue(self.md["mimo.llm.attention.qkv_bias"])
        self.assertFalse(self.md["mimo.llm.attention.qk_norm"])
        self.assertEqual(self.md["openasr.model.family"], "mimo-asr")
        self.assertEqual(self.md["openasr.pack.quant"], "q8_0")


# ---------------------------------------------------------------------------
# tokenizer baking (vocab.json + merges.txt + added_tokens_decoder)
# ---------------------------------------------------------------------------

class TokenizerLoadTest(unittest.TestCase):
    def _write_tokenizer_fixture(self, root: Path):
        (root / "vocab.json").write_text(json.dumps({"a": 0, "b": 1, "ab": 2}))
        (root / "merges.txt").write_text("#version: fake\na b\n")
        (root / "tokenizer_config.json").write_text(json.dumps({
            "added_tokens_decoder": {
                "3": {"content": "<|endoftext|>"},
                "5": {"content": "<|im_start|>"},
            }
        }))

    def test_load_vocab_tokens_orders_by_id(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_tokenizer_fixture(root)
            tokens = C.load_vocab_tokens(root)
            self.assertEqual(tokens, ["a", "b", "ab"])

    def test_load_merges_strips_comments_and_blank_lines(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_tokenizer_fixture(root)
            self.assertEqual(C.load_merges(root), ["a b"])

    def test_patch_added_tokens_extends_and_fills_gap(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_tokenizer_fixture(root)
            tokens = C.patch_added_tokens(root, ["a", "b", "ab"])
            # id 3 (<|endoftext|>) and id 5 (<|im_start|>) patched in; id 4
            # (the gap) becomes an empty placeholder.
            self.assertEqual(tokens, ["a", "b", "ab", "<|endoftext|>", "", "<|im_start|>"])

    def test_load_tokenizer_pads_to_vocab_size(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_tokenizer_fixture(root)
            tokens, merges = C.load_tokenizer(root, vocab_size=10)
            self.assertEqual(len(tokens), 10)
            self.assertEqual(tokens[:6], ["a", "b", "ab", "<|endoftext|>", "", "<|im_start|>"])
            self.assertEqual(tokens[6:], [""] * 4)
            self.assertEqual(merges, ["a b"])


# ---------------------------------------------------------------------------
# full tiny synthetic round-trip
# ---------------------------------------------------------------------------

def _tiny_configs():
    main_cfg = {
        "hidden_size": 8, "num_hidden_layers": 2, "num_attention_heads": 2,
        "num_key_value_heads": 1, "head_dim": 4, "intermediate_size": 16,
        "rope_theta": 640000, "rms_norm_eps": 1e-6, "vocab_size": 32,
        "max_position_embeddings": 64, "attention_bias": True,
        "audio_channels": 8, "group_size": 4,
        "input_local_layers": 1, "input_local_dim": 4, "input_full_attention": True,
        "speech_vocab_size": "5-5-3-3-3-3-3-3", "speech_zeroemb_idx": "4-4-2-2-2-2-2-2",
        "audio_config": {
            "input_local_attn_heads": 2, "input_local_head_dim": 2,
            "input_local_intermediate_size": 8, "rope_theta": 640000,
        },
    }
    tok_cfg = {
        "n_mels": 4, "d_model": 6, "encoder_layers": 3, "encoder_attention_heads": 2,
        "encoder_ffn_dim": 8, "encoder_skip_layer_id": 3, "kernel_size": 3,
        "stride_size": 2, "avg_pooler": 2, "rope_theta": 10000,
        "sampling_rate": 24000, "nfft": 8, "hop_length": 2, "window_size": 8,
        "fmin": 0, "fmax": None, "num_quantizers": 4, "codebook_size": [5, 5, 3, 3],
    }
    return main_cfg, tok_cfg


def _tiny_hparams():
    m, t = _tiny_configs()
    return C.MainHParams.from_config(m), C.TokHParams.from_config(t)


def _write_tiny_sources(root: Path):
    import torch
    from safetensors.torch import save_file

    main_cfg, tok_cfg = _tiny_configs()
    main_dir = root / "main"
    main_dir.mkdir()
    tok_dir = root / "tok"
    tok_dir.mkdir()
    (main_dir / "config.json").write_text(json.dumps(main_cfg))
    (tok_dir / "config.json").write_text(json.dumps(tok_cfg))

    H, L, KV, FF, V = 8, 2, 4, 16, 32
    ILD, ILFF = 4, 8

    def r(*shape):
        return torch.randn(*shape, dtype=torch.float32)

    main = {"model.embed_tokens.weight": r(V, H), "model.norm.weight": r(H), "lm_head.weight": r(V, H)}
    for i in range(L):
        p = f"model.layers.{i}."
        main[p + "input_layernorm.weight"] = r(H)
        main[p + "self_attn.q_proj.weight"] = r(H, H)
        main[p + "self_attn.q_proj.bias"] = r(H)
        main[p + "self_attn.k_proj.weight"] = r(KV, H)
        main[p + "self_attn.k_proj.bias"] = r(KV)
        main[p + "self_attn.v_proj.weight"] = r(KV, H)
        main[p + "self_attn.v_proj.bias"] = r(KV)
        main[p + "self_attn.o_proj.weight"] = r(H, H)
        main[p + "post_attention_layernorm.weight"] = r(H)
        main[p + "mlp.gate_proj.weight"] = r(FF, H)
        main[p + "mlp.up_proj.weight"] = r(FF, H)
        main[p + "mlp.down_proj.weight"] = r(H, FF)
    vocabs = [5, 5, 3, 3, 3, 3, 3, 3]
    for n, vv in enumerate(vocabs):
        main[f"speech_embeddings.{n}.weight"] = r(vv, ILD)
    main["speech_group_downcast.weight"] = r(H, ILD * 4)
    # input-local (1 layer)
    p = "input_local_transformer.layers.0."
    main[p + "input_layernorm.weight"] = r(ILD)
    main[p + "self_attn.q_proj.weight"] = r(ILD, ILD)
    main[p + "self_attn.q_proj.bias"] = r(ILD)
    main[p + "self_attn.k_proj.weight"] = r(ILD, ILD)
    main[p + "self_attn.k_proj.bias"] = r(ILD)
    main[p + "self_attn.v_proj.weight"] = r(ILD, ILD)
    main[p + "self_attn.v_proj.bias"] = r(ILD)
    main[p + "self_attn.o_proj.weight"] = r(ILD, ILD)
    main[p + "post_attention_layernorm.weight"] = r(ILD)
    main[p + "mlp.gate_proj.weight"] = r(ILFF, ILD)
    main[p + "mlp.up_proj.weight"] = r(ILFF, ILD)
    main[p + "mlp.down_proj.weight"] = r(ILD, ILFF)
    main["input_local_transformer.norm.weight"] = r(ILD)
    # discardable stacks
    main["local_transformer.layers.0.input_layernorm.weight"] = r(ILD)
    main["local_transformer.norm.weight"] = r(ILD)
    main["local_transformer_lm_heads.0.weight"] = r(5, ILD)
    main["hidden_states_downcast.weight"] = r(ILD, H)
    save_file(main, str(main_dir / "model-00001-of-00001.safetensors"))

    D, EL, EFF, NM = 6, 3, 8, 4
    tok = {
        "encoder.conv1.weight": r(D, NM, 3), "encoder.conv1.bias": r(D),
        "encoder.conv2.weight": r(D, D, 3), "encoder.conv2.bias": r(D),
        "encoder.layer_norm.weight": r(D), "encoder.layer_norm.bias": r(D),
        "encoder.down_sample_layer.0.weight": r(D, D, 2),
        "encoder.down_sample_norm.weight": r(D), "encoder.down_sample_norm.bias": r(D),
    }
    for i in range(EL):
        p = f"encoder.layers.{i}."
        tok[p + "self_attn.q_proj.weight"] = r(D, D)
        tok[p + "self_attn.q_proj.bias"] = r(D)
        tok[p + "self_attn.k_proj.weight"] = r(D, D)  # no k bias
        tok[p + "self_attn.v_proj.weight"] = r(D, D)
        tok[p + "self_attn.v_proj.bias"] = r(D)
        tok[p + "self_attn.out_proj.weight"] = r(D, D)
        tok[p + "self_attn.out_proj.bias"] = r(D)
        tok[p + "self_attn_layer_norm.weight"] = r(D)
        tok[p + "self_attn_layer_norm.bias"] = r(D)
        tok[p + "final_layer_norm.weight"] = r(D)
        tok[p + "final_layer_norm.bias"] = r(D)
        tok[p + "fc1.weight"] = r(EFF, D)
        tok[p + "fc1.bias"] = r(EFF)
        tok[p + "fc2.weight"] = r(D, EFF)
        tok[p + "fc2.bias"] = r(D)
    cb_sizes = [5, 5, 3, 3]
    for q, cs in enumerate(cb_sizes):
        p = f"encoder.quantizer.vq.layers.{q}._codebook."
        tok[p + "embed"] = r(cs, D)
        tok[p + "embed_avg"] = r(cs, D)
        tok[p + "cluster_size"] = r(cs)
        tok[p + "inited"] = torch.ones(1, dtype=torch.float32)
    # discardable decoder tensors
    tok["decoder.layer_norm.weight"] = r(D)
    tok["decoder.layers.0.fc1.weight"] = r(EFF, D)
    save_file(tok, str(tok_dir / "model.safetensors"))

    # Tiny tokenizer fixture (vocab_size=32 to match main_cfg above): a
    # handful of real vocab entries plus the ChatML/audio special tokens
    # patched in via added_tokens_decoder, padded to 32 by load_tokenizer.
    (main_dir / "vocab.json").write_text(json.dumps({
        "a": 0, "b": 1, "ab": 2, "user": 3, "assistant": 4,
    }))
    (main_dir / "merges.txt").write_text("#version\na b\n")
    (main_dir / "tokenizer_config.json").write_text(json.dumps({
        "added_tokens_decoder": {
            "5": {"content": "<|endoftext|>"},
            "6": {"content": "<|im_start|>"},
            "7": {"content": "<|im_end|>"},
            "17": {"content": "<|sosp|>"},
            "18": {"content": "<|eosp|>"},
            "19": {"content": "<|empty|>"},
        }
    }))

    return main_dir, tok_dir / "model.safetensors", main, tok


class RoundTripTest(unittest.TestCase):
    def test_tiny_end_to_end(self):
        import gguf

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            main_dir, tok_path, main_src, tok_src = _write_tiny_sources(root)
            out = root / "mimo-tiny-q8_0.oasr"
            res = C.write_pack(main_dir, tok_path, out, "q8_0", "mimo-tiny-q8_0", verbose=False)
            self.assertTrue(out.exists())
            self.assertGreater(res["tensor_count"], 0)

            reader = gguf.GGUFReader(str(out))
            names = {t.name for t in reader.tensors}
            by_name = {t.name: t for t in reader.tensors}

            # discarded stacks absent
            for absent in [
                "local_transformer.norm.weight", "hidden_states_downcast.weight",
                "audiotok.quant.0.codebook_avg",
            ]:
                self.assertNotIn(absent, names)
            self.assertFalse(any("local_transformer" in n for n in names))
            self.assertFalse(any(n.startswith("decoder") for n in names))
            self.assertFalse(any("embed_avg" in n or "cluster_size" in n or "inited" in n for n in names))

            # backbone present
            for want in [
                "token_embd.weight", "output.weight", "output_norm.weight",
                "blk.0.attn_q.weight", "blk.0.attn_q.bias", "blk.0.attn_k.bias",
                "blk.1.ffn_gate.weight", "blk.1.ffn_norm.weight",
                "speech_group_proj.weight",
                "inlocal.blk.0.attn_v.bias", "inlocal.norm.weight",
                "audiotok.conv1.weight", "audiotok.conv2.weight",
                "audiotok.blk.0.attn_q.bias", "audiotok.blk.2.ffn_down.weight",
                "audiotok.norm.weight", "audiotok.down_sample.weight",
                "audiotok.down_sample_norm.weight",
                "audiotok.mel_filters", "audiotok.mel_window",
            ]:
                self.assertIn(want, names, want)

            # exactly 8 speech_embd tables + 4 packed codebooks
            self.assertEqual(sum(1 for n in names if n.startswith("speech_embd.")), 8)
            self.assertEqual(sum(1 for n in names if n.startswith("audiotok.quant.")), 4)

            # tokenizer k_proj has no bias
            self.assertNotIn("audiotok.blk.0.attn_k.bias", names)

            # --- shape checks (GGUF stores dims reversed vs torch) ---
            # speech_embd.0: torch [5,4] -> ggml logical [4,5]
            self.assertEqual([int(x) for x in by_name["speech_embd.0.weight"].shape], [4, 5])
            # codebook 0: torch [5,6] -> [6,5]
            self.assertEqual([int(x) for x in by_name["audiotok.quant.0.codebook"].shape], [6, 5])
            # conv1: torch [6,4,3] -> [3,4,6]
            self.assertEqual([int(x) for x in by_name["audiotok.conv1.weight"].shape], [3, 4, 6])

            # --- value fidelity (reader returns data in original torch shape) ---
            # codebook stored f32 -> bit-exact
            cb = _tensor_data(reader, "audiotok.quant.0.codebook")  # [vocab,dim]
            np.testing.assert_allclose(cb, tok_src["encoder.quantizer.vq.layers.0._codebook.embed"].numpy(), rtol=0, atol=0)
            # speech_embd stored f16 -> within f16 tolerance
            se = _tensor_data(reader, "speech_embd.3.weight")  # [vocab,dim]
            np.testing.assert_allclose(se, main_src["speech_embeddings.3.weight"].numpy(), rtol=0, atol=2e-3)

            # --- metadata blood lessons in the actual file ---
            self.assertEqual(_kv_u32(reader, "mimo.tok.encoder.skip_layer_id"), 3)
            self.assertEqual(_kv_u32(reader, "mimo.tok.conv1.stride"), 1)
            self.assertEqual(_kv_u32(reader, "mimo.tok.conv2.stride"), 2)
            self.assertEqual(_kv_u32(reader, "mimo.audio.channels"), 8)
            self.assertEqual(_kv_u32(reader, "mimo.llm.block_count"), 2)

            # skip lesson: captured layer (skip_id-1=idx2) AND final encoder layer present
            skip_id = _kv_u32(reader, "mimo.tok.encoder.skip_layer_id")
            block_count = _kv_u32(reader, "mimo.tok.block_count")
            self.assertIn(f"audiotok.blk.{skip_id - 1}.ffn_norm.weight", names)
            self.assertIn(f"audiotok.blk.{block_count - 1}.ffn_norm.weight", names)

            # tokenizer baked from the tiny vocab.json/merges.txt/tokenizer_config.json fixture
            self.assertEqual(_kv_str(reader, "tokenizer.ggml.model"), "gpt2")
            tokens = _kv_str_array(reader, "tokenizer.ggml.tokens")
            self.assertEqual(len(tokens), 32)  # padded to main_cfg's vocab_size
            self.assertEqual(tokens[:8], ["a", "b", "ab", "user", "assistant",
                                           "<|endoftext|>", "<|im_start|>", "<|im_end|>"])
            self.assertEqual(tokens[17:20], ["<|sosp|>", "<|eosp|>", "<|empty|>"])
            self.assertEqual(_kv_str_array(reader, "tokenizer.ggml.merges"), ["a b"])

    def test_fp16_pack_has_no_quantized_tensors(self):
        import gguf

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            main_dir, tok_path, _, _ = _write_tiny_sources(root)
            out = root / "mimo-tiny-fp16.oasr"
            C.write_pack(main_dir, tok_path, out, "fp16", "mimo-tiny-fp16", verbose=False)
            reader = gguf.GGUFReader(str(out))
            for t in reader.tensors:
                self.assertIn(t.tensor_type.name, ("F16", "F32"), f"{t.name}={t.tensor_type.name}")


def _tensor_data(reader, name):
    for t in reader.tensors:
        if t.name == name:
            return np.array(t.data)
    raise KeyError(name)


def _kv_u32(reader, key):
    field = reader.get_field(key)
    return int(field.parts[field.data[-1]][0])


def _kv_str(reader, key):
    field = reader.get_field(key)
    part = field.parts[field.data[-1]]
    return bytes(part).decode("utf-8") if not isinstance(part, str) else part


def _kv_str_array(reader, key):
    field = reader.get_field(key)
    out = []
    for idx in field.data:
        part = field.parts[idx]
        out.append(bytes(part).decode("utf-8") if not isinstance(part, str) else part)
    return out


if __name__ == "__main__":
    unittest.main(verbosity=2)
