[English](README.md) | [简体中文](README.zh-CN.md)

<div align="center">

# OpenASR

**语音转文字,完全在你自己的设备上运行。**

[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![CI](https://github.com/QuintinShaw/openasr/actions/workflows/ci.yml/badge.svg)](https://github.com/QuintinShaw/openasr/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/QuintinShaw/openasr)](https://github.com/QuintinShaw/openasr/releases)
[![Downloads](https://img.shields.io/github/downloads/QuintinShaw/openasr/total)](https://github.com/QuintinShaw/openasr/releases)

[官网](https://openasr.org) · [文档](docs/DOCS_INDEX.md) · [License](LICENSE)

<img src="https://openasr.org/assets/openasr-desktop-preview-zh.gif" alt="OpenASR 桌面应用" width="720" />

<sub>v1 之前的早期阶段,正在活跃开发中。0.x 版本间的命令行参数、API 和包格式可能调整。</sub>

</div>

---

<div align="center">
<h3><a href="https://openasr.org/zh/download/">下载桌面应用</a></h3>
<p><strong>macOS</strong> (Apple Silicon) · <strong>Windows</strong> (x64, Windows 10+) · Linux 桌面版开发中</p>
</div>

不需要命令行,不需要配环境。安装应用,拖入音频文件,转写结果直接出来——默认全程本地运行。

> 本仓库是桌面应用背后的 Apache-2.0 开源核心:Rust 命令行工具 + 本地 OpenAI 兼容 HTTP API + ggml 推理引擎。桌面应用在同一个引擎之上套了原生图形界面,没有任何隐藏的网络通道。

---

## 它能做什么

- **音频转文字** — 单个文件或整个文件夹,输出纯文本、SRT/VTT 字幕,或带逐字时间戳的 JSON
- **实时听写** — 连上麦克风,边说边出字,流式部分结果实时刷新
- **系统音频捕获** — 会议、网课、播客——直接转写电脑正在播放的声音
- **说话人分离** — 自动标注"谁在说话"
- **翻译** — 转录的同时翻译成英文,一步到位
- **本地 API** — 兼容 OpenAI `/v1/audio/transcriptions`,现有 SDK 直接对接

## 为什么选 OpenASR

**隐私。** 默认本地模式下,音频留在你的设备上。远程算力仅在你显式配对并启用后可用,详见 [SECURITY.md](SECURITY.md#local-first-security-notes)。没有遥测、没有静默上传、没有静默联网回退。引擎要么给你一份真实的转写结果,要么明确告诉你哪里出了问题。

**广度。** 11 个模型家族、26 个公开模型——Whisper、Qwen3-ASR、Parakeet、SenseVoice、FireRed、Dolphin、Moonshine……选对模型比选对工具更重要,而 OpenASR 把它们统一到一个运行时里,CPU 和 Apple Metal 都能跑。

**开源。** 引擎代码 Apache-2.0。每个模型包的许可证以 registry 条目和 pack metadata 为准。每次模型下载都经过签名目录的完整性校验,装到本地的就是发布者打包的原件。

---

## 开发者入口

### CLI 快速上手

```bash
# 方式一: Homebrew (macOS / Linux)
brew install quintinshaw/tap/openasr

# 方式二: 一行安装脚本 (macOS / Linux)
curl -fsSL https://dl.openasr.org/install.sh | sh

# 方式三: 从 Releases 页面下载预编译二进制
# https://github.com/QuintinShaw/openasr/releases

# 转写一个文件 (首次运行会提示下载默认模型,需要你确认)
openasr transcribe recording.wav

# 麦克风实时听写
openasr live

# SRT 字幕 + 说话人分离
openasr transcribe meeting.wav -f srt --diarize
```

详细步骤见 [Quickstart](docs/QUICKSTART.md),或 `openasr --help` 查看完整命令。

### 本地 API

```bash
openasr serve

curl http://127.0.0.1:8080/v1/audio/transcriptions \
  -F file=@audio.wav -F model=qwen3-asr-0.6b
```

与 OpenAI SDK 直接兼容(`base_url="http://127.0.0.1:8080/v1"`)。API Key 和 Agent 集成见 [Agent Integration](docs/AGENT_INTEGRATION.md)。

### 从源码构建

```bash
git clone --recurse-submodules https://github.com/QuintinShaw/openasr.git
cd openasr
cargo build --release -p openasr-cli
```

需要 Rust(`rust-toolchain.toml` 锁定版本)、CMake 和 C/C++ 工具链。完整环境搭建见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 模型

11 个家族、26 个公开模型:从跑得比实时快几倍的小型英文模型,到覆盖 100 多种语言的大型多语言模型。在 [openasr.org/models](https://openasr.org/models/) 浏览全部模型,或在命令行里查看:

```bash
openasr search              # 浏览可用模型
openasr pull whisper-small    # 安装一个试试
```

性能基准数据见 [Performance](perf/PERFORMANCE.md)。

## 文档

| | |
|---|---|
| [文档索引](docs/DOCS_INDEX.md) | 所有文档的导航页 |
| [Quickstart](docs/QUICKSTART.md) | 三条命令完成第一次转写 |
| [FAQ](docs/FAQ.md) | 常见问题 |
| [已知限制](docs/KNOWN_LIMITATIONS.md) | 当前能做和暂时做不到的 |
| [路线图](docs/ROADMAP.md) | 接下来要做什么 |
| [架构](ARCHITECTURE.md) | crate 关系与转写流水线 |

## 参与贡献

欢迎贡献。分支命名、PR 规范、DCO 签名等开发流程见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 许可证

[Apache License 2.0](LICENSE),详见 [NOTICE](NOTICE)。

ggml 推理后端为 MIT 许可。每个模型包的许可证以 registry 条目和 pack metadata 为准;可能包含 Apache-2.0、MIT、CC-BY、FunASR 或其他上游条款。这不是穷尽保证。致谢完整列表见 [ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md)。

## 找到我们

OpenASR 还在早期，「早鸟营」微信群是离我们最近的地方——聊用法、反馈问题、第一时间拿到新版本。

<p align="center">
  <img src="https://openasr.org/assets/wechat-group.png" width="240" alt="OpenASR 早鸟营微信群二维码">
</p>

<sub>二维码每几天更新一次。如果扫码显示过期，来 [Issues](../../issues) 说一声，我们会尽快换新。</sub>
