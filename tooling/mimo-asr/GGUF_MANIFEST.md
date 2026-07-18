# MiMo-V2.5-ASR .oasr GGUF manifest (for P2.2 runtime)
# generated from mimo-v2.5-asr-q8_0.oasr by tooling/mimo-asr/convert_mimo_asr.py

## Metadata keys (key = value)
GGUF.version = 3
GGUF.tensor_count = 1016
GGUF.kv_count = 72
general.architecture = mimo-asr
openasr.package.version = 1
openasr.model.family = mimo-asr
openasr.model.architecture = mimo-asr
openasr.model.id = mimo-v2.5-asr-q8_0
openasr.audio.frontend = mimo-tokenizer-rvq-v0
openasr.decode.policy = mimo-asr.greedy.seq2seq.v0
openasr.pack.quant = q8_0
tokenizer.ggml.model = gpt2
tokenizer.ggml.tokens = [151680 strings, official Qwen2 BPE vocab + MiMo's added special tokens]
tokenizer.ggml.merges = [151291 "a b" byte-pair merge rules]
mimo.llm.block_count = 36
mimo.llm.context_length = 8192
mimo.llm.embedding_length = 4096
mimo.llm.feed_forward_length = 11008
mimo.llm.attention.head_count = 32
mimo.llm.attention.head_count_kv = 8
mimo.llm.attention.key_length = 128
mimo.llm.attention.value_length = 128
mimo.llm.attention.layer_norm_rms_epsilon = 9.999999974752427e-07
mimo.llm.rope.freq_base = 640000.0
mimo.llm.vocab_size = 151680
mimo.llm.attention.qkv_bias = True
mimo.llm.attention.qk_norm = False
mimo.audio.channels = 8
mimo.audio.group_size = 4
mimo.inlocal.block_count = 6
mimo.inlocal.embedding_length = 1024
mimo.inlocal.attention.head_count = 64
mimo.inlocal.attention.head_dim = 16
mimo.inlocal.feed_forward_length = 4096
mimo.inlocal.full_attention = True
mimo.inlocal.rope.freq_base = 640000.0
mimo.speech.vocab_size = [1025, 1025, 129, 129, 129, 129, 129, 129]
mimo.speech.zeroemb_idx = [1024, 1024, 128, 128, 128, 128, 128, 128]
mimo.tok.block_count = 32
mimo.tok.embedding_length = 1280
mimo.tok.attention.head_count = 20
mimo.tok.feed_forward_length = 5120
mimo.tok.encoder.skip_layer_id = 3
mimo.tok.conv.kernel_size = 3
mimo.tok.conv1.stride = 1
mimo.tok.conv2.stride = 2
mimo.tok.down_sample.stride = 2
mimo.tok.rope.freq_base = 10000.0
mimo.tok.ln_type = layernorm
mimo.tok.attention.qk_bias_asymmetric = True
mimo.tok.rvq.num_quantizers_total = 20
mimo.tok.rvq.num_quantizers_packed = 8
mimo.tok.rvq.codebook_sizes = [1024, 1024, 128, 128, 128, 128, 128, 128]
mimo.mel.sample_rate = 24000
mimo.mel.n_fft = 960
mimo.mel.hop_length = 240
mimo.mel.win_length = 960
mimo.mel.n_mels = 128
mimo.mel.f_min = 0.0
mimo.mel.f_max = 12000.0
mimo.mel.mel_scale = htk
mimo.mel.norm = none
mimo.mel.power = 1.0
mimo.mel.log_type = ln
mimo.mel.log_clip = 1.0000000116860974e-07
mimo.mel.center = True
mimo.special.eos_id = 151643
mimo.special.im_start_id = 151644
mimo.special.im_end_id = 151645
mimo.special.sosp_id = 151665
mimo.special.eosp_id = 151666
mimo.special.empty_id = 151667
mimo.special.sostm_id = 151670
mimo.special.eostm_id = 151671
mimo.special.eot_id = 151672

## Tensors (collapsed by layer index): gguf_name  ggml_shape  type   -- count
audiotok.blk.{i}.attn_k.weight           [1280, 1280]       F16    -- 32
audiotok.blk.{i}.attn_norm.bias          [1280]             F32    -- 32
audiotok.blk.{i}.attn_norm.weight        [1280]             F32    -- 32
audiotok.blk.{i}.attn_out.bias           [1280]             F32    -- 32
audiotok.blk.{i}.attn_out.weight         [1280, 1280]       F16    -- 32
audiotok.blk.{i}.attn_q.bias             [1280]             F32    -- 32
audiotok.blk.{i}.attn_q.weight           [1280, 1280]       F16    -- 32
audiotok.blk.{i}.attn_v.bias             [1280]             F32    -- 32
audiotok.blk.{i}.attn_v.weight           [1280, 1280]       F16    -- 32
audiotok.blk.{i}.ffn_down.bias           [1280]             F32    -- 32
audiotok.blk.{i}.ffn_down.weight         [5120, 1280]       F16    -- 32
audiotok.blk.{i}.ffn_norm.bias           [1280]             F32    -- 32
audiotok.blk.{i}.ffn_norm.weight         [1280]             F32    -- 32
audiotok.blk.{i}.ffn_up.bias             [5120]             F32    -- 32
audiotok.blk.{i}.ffn_up.weight           [1280, 5120]       F16    -- 32
audiotok.conv1.bias                      [1280]             F32    -- 1
audiotok.conv1.weight                    [3, 128, 1280]     F16    -- 1
audiotok.conv2.bias                      [1280]             F32    -- 1
audiotok.conv2.weight                    [3, 1280, 1280]    F16    -- 1
audiotok.down_sample.weight              [2, 1280, 1280]    F16    -- 1
audiotok.down_sample_norm.bias           [1280]             F32    -- 1
audiotok.down_sample_norm.weight         [1280]             F32    -- 1
audiotok.mel_filters                     [128, 481]         F32    -- 1
audiotok.mel_window                      [960]              F32    -- 1
audiotok.norm.bias                       [1280]             F32    -- 1
audiotok.norm.weight                     [1280]             F32    -- 1
audiotok.quant.{i}.codebook              [1280, 1024]       F32    -- 8
blk.{i}.attn_k.bias                      [1024]             F32    -- 36
blk.{i}.attn_k.weight                    [4096, 1024]       Q8_0   -- 36
blk.{i}.attn_norm.weight                 [4096]             F32    -- 36
blk.{i}.attn_output.weight               [4096, 4096]       Q8_0   -- 36
blk.{i}.attn_q.bias                      [4096]             F32    -- 36
blk.{i}.attn_q.weight                    [4096, 4096]       Q8_0   -- 36
blk.{i}.attn_v.bias                      [1024]             F32    -- 36
blk.{i}.attn_v.weight                    [4096, 1024]       Q8_0   -- 36
blk.{i}.ffn_down.weight                  [11008, 4096]      Q8_0   -- 36
blk.{i}.ffn_gate.weight                  [4096, 11008]      Q8_0   -- 36
blk.{i}.ffn_norm.weight                  [4096]             F32    -- 36
blk.{i}.ffn_up.weight                    [4096, 11008]      Q8_0   -- 36
inlocal.blk.{i}.attn_k.bias              [1024]             F32    -- 6
inlocal.blk.{i}.attn_k.weight            [1024, 1024]       F16    -- 6
inlocal.blk.{i}.attn_norm.weight         [1024]             F32    -- 6
inlocal.blk.{i}.attn_output.weight       [1024, 1024]       F16    -- 6
inlocal.blk.{i}.attn_q.bias              [1024]             F32    -- 6
inlocal.blk.{i}.attn_q.weight            [1024, 1024]       F16    -- 6
inlocal.blk.{i}.attn_v.bias              [1024]             F32    -- 6
inlocal.blk.{i}.attn_v.weight            [1024, 1024]       F16    -- 6
inlocal.blk.{i}.ffn_down.weight          [4096, 1024]       F16    -- 6
inlocal.blk.{i}.ffn_gate.weight          [1024, 4096]       F16    -- 6
inlocal.blk.{i}.ffn_norm.weight          [1024]             F32    -- 6
inlocal.blk.{i}.ffn_up.weight            [1024, 4096]       F16    -- 6
inlocal.norm.weight                      [1024]             F32    -- 1
output.weight                            [4096, 151680]     Q8_0   -- 1
output_norm.weight                       [4096]             F32    -- 1
speech_embd.{i}.weight                   [1024, 1025]       F16    -- 8
speech_group_proj.weight                 [4096, 4096]       F16    -- 1
token_embd.weight                        [4096, 151680]     Q8_0   -- 1

## total tensors = 1016
