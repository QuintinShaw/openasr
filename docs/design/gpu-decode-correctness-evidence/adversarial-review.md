# GPU decode correctness 合同对抗审查

审查对象：`docs/design/gpu-decode-correctness-contract.md` 第 18 问。
代码权威：`docs/gpu-top1-correctness-design` @ `90aceda2f`（含 Metal SWOOSH pin、Qwen FreshGraph host KV、seq2seq token steps）。
口径：对问题中的肯定命题作答。缺主机不是通过。

证据文件（同目录）：`hardware-unavailable.txt`、`desktop-plugin-switch.fail.txt`。
门测试：`tooling/release-manifest/gpu_correctness_gate_test.py`、`gpu_decode_fail_closed_evidence_test.py`。

---

## 1. ordinary ARGMAX 是否被正确描述为无 portable tie contract 的后端 reduction

**结论：成立。** CPU last-max / first-max 是不同算子；CUDA ordinary ARGMAX 是 reduction 顺序；Metal `kernel_argmax_f32` 不是 portable last-max。规划器 `supports_argmax_first` 只是 `supports_op` 声明。本机未跑 CUDA/Vulkan/HIP `test-backend-ops`（见 `hardware-unavailable.txt`）。

## 2. 是否还有生产 reverse-selector 落在清单外

**结论：不成立。** 生产无 `top1_argmax_first_max_reversed`。选择器是 `top1_argmax_first_max`。Qwen 生产 fused top1 `cfg(not(test))` 恒 false。XASR host last-max；MiMo RVQ host first-max。`native_first_max_compact_is_proven` 只认 CPU。

## 3. logits 消费者激活时能否合法消费 device top1

**结论：不成立（规划器会强制完整输出）。** `GgmlDecodeLogitsConsumers` 的 phrase_bias / timestamps / suppression / debug_logits / host_visible 迫使 FullLogits。无独立 probabilities 字段；那是残留缝，不是当前请求面的 bypass。

## 4. 无 tie 时现场重复能否发生；四象限是否定位现场 CUDA 第一分叉

**结论：前半成立；Windows CUDA 仍证据不足。** encoder/KV/kernel 可在无并列最大值时产生重复。旧 `firered-four-quadrant.json` 只能定位到 aggregate `subsample_out`；完整 12-tap 的 `firered-encoder-stem-m1-q4-jfk.json` 已把本机 CPU/Metal 第一 checksum 分叉收窄到 `subsample_input` / `mel_4d`，即第一层 convolution 之前。它仍是 M1 q4_k，不是 Windows CUDA fp16，也不能单凭 checksum 区分输入上传与读回边界。`hardware-unavailable.txt` 记录 CUDA 不可用。不得把 Metal/CPU 当 CUDA 通过。

## 5. dual-output 会不会掩盖缺陷

**结论：成立（风险真实；已禁止用它授权生产 compact）。** `authorizes_production_compact` 恒 false。

## 6. fresh/reuse 是否独立 KV

**结论：合成探针成立；GPU 真模型不足。** 两个 `GgmlCpuGraphRunner`。生产 reuse 证据 Unknown → FreshGraph。无 CUDA/HIP 双 runtime。

## 7. `supports_op` 是否冒充 persistent graph / 真模型证据

**结论：不成立。** compact 需要 `supports_op && proven`；proven 仅 CPU。假 GPU `supports_op` 仍 FullLogits。reuse 门只认 `ReusableGraph`。

## 8. Metal 是否祖父 reverse

**结论：不成立。** reverse 已删；Metal 走 FullLogits。本 HEAD Layer-3：FireRed/Qwen/XASR Metal 收据存在且 compute 仅 MTL0，不是 reverse 路径。Metal 仍无 native ARGMAX_FIRST。

## 9. HIP capture-on 是否在证明前关闭 compact/reuse

**结论：后半成立；前半证据不足。** HIP 与 CUDA 一样 FullLogits+FreshGraph。无 HIP 主机，capture-on 刷新未测。见 `hardware-unavailable.txt`。

## 10. owner 跨 plan 复用

**结论：点名的 SenseVoice/Qwen 不成立（key 含 plan）。** Whisper/Dolphin/CTC-TDT 拓扑不随 plan 变，测试证明同一 key 服务两种 plan。

## 11. 是否仅凭 placement 宣称家族正确

**结论：门实现上不成立；CUDA/Vulkan/HIP token 格仍空。** `validate_matrix` 要求 placement 与 token_transcript 分格。无收据则 `require_activation` 拒绝 Auto/explicit。不得用 CPU/Metal 收据填 GPU 格。

## 12. Untested/Deferred 时能否 Auto

**结论：门会拒；活审计文档仍可能写 Untested。** `test_cuda_vulkan_hip_without_receipts_cannot_auto_or_explicit`、`test_public_generation_blocks_untested_advertised_gpu_lanes`。缺收据 = 不可选，不是 skip。

## 13. 收据未齐能否激活公开 catalog

**结论：编排意图上不成立。** `gpu_correctness_gate.py validate` 在 finalize / deploy-catalog 之前。无绑定收据则失败。本机无完整 GPU 矩阵，因此发布会被挡住，不是被放行。

## 14. 桌面插件能否在 daemon 转录前报成功

**结论：本机产品 E2E 未跑通，记 FAIL 不是 skip。** `desktop-plugin-switch.fail.txt`：`result=FAIL` `skipped=false` `host_mode=legacy_static`。shipped `require_desktop_plugin_switch` 将 FAIL 视为不可选。

## 15. 删 reverse 是否让无 native first-max 的后端回归

**结论：不成立。** 未证明车道 FullLogits，保留 FullDevice，不静默 CPU。测试 `unproven_gpu_lanes_keep_full_device_and_complete_outputs`。

## 16. 工作量是否漏 XASR/RVQ/SenseVoice/HIP/receipt/E2E

**结论：不成立（合同已列入）。** 文档覆盖不等于 CUDA/E2E 已完成。

## 17. 收据隐私 / error-string 旁路

**结论：error-string 旁路不成立；生产路径用类型化 plan。** Layer-3 命令行仍可能含本地路径，不得当政策通道。

## 18. output-plan attestation 失败是否保留旧 runtime

**结论：shipped 入口成立。** `activate_runtime_with_output_plan` + `shipped_activation_rejects_mismatched_output_plan_and_keeps_previous_runtime`。HTTP 层 selected 与 staged 目前同一次 resolve，mismatch 覆盖在该入口。

---

## 未闭合（不是通过）

1. Windows CUDA 现场第一分叉未定位。
2. CUDA / 物理 Vulkan / HIP 三层真包收据为零；格子不可激活。
3. desktop 插件切换 5 步产品路径未跑通（`legacy_static` FAIL）。
4. HIP capture-on 持久图未测。
5. 公开 catalog 活审计仍可能同时广告 GPU 与 Untested；`--public` 会拒。
