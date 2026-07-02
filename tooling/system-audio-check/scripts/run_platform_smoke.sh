#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

case "$(uname -s)" in
  Darwin)
    command -v afplay >/dev/null 2>&1 || {
      echo "macOS system-audio smoke requires afplay." >&2
      exit 1
    }
    TEST_NAME="macos_core_audio_system_audio_smoke_emits_non_silent_frames"
    ;;
  Linux)
    missing=()
    command -v pactl >/dev/null 2>&1 || missing+=("pactl")
    command -v parec >/dev/null 2>&1 || missing+=("parec")
    if ! command -v paplay >/dev/null 2>&1 \
      && ! command -v pw-play >/dev/null 2>&1 \
      && ! command -v aplay >/dev/null 2>&1; then
      missing+=("paplay|pw-play|aplay")
    fi
    if ((${#missing[@]} > 0)); then
      echo "Linux system-audio smoke is missing command(s): ${missing[*]}" >&2
      echo "Install PulseAudio/PipeWire Pulse tools and one playback command, then rerun." >&2
      exit 1
    fi
    TEST_NAME="linux_pulse_monitor_system_audio_smoke_emits_non_silent_frames"
    ;;
  *)
    echo "Unsupported OS for this script. Use run_platform_smoke.ps1 on Windows." >&2
    exit 1
    ;;
esac

cd "${REPO_ROOT}"
exec cargo test --manifest-path crates/openasr-system-audio/Cargo.toml "${TEST_NAME}" -- --ignored --nocapture
