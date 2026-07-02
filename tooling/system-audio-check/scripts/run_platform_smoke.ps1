param(
    [switch]$Help
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($Help) {
    Write-Output "Runs the Windows WASAPI system-audio smoke test with local playback."
    Write-Output "Usage: pwsh tooling/system-audio-check/scripts/run_platform_smoke.ps1"
    exit 0
}

$isWindowsPlatformVariable = Get-Variable -Name IsWindows -ErrorAction SilentlyContinue
$isWindowsPlatform = if ($isWindowsPlatformVariable) {
    $IsWindows
} else {
    [System.Environment]::OSVersion.Platform -eq [System.PlatformID]::Win32NT
}

if (-not $isWindowsPlatform) {
    throw "run_platform_smoke.ps1 must run on Windows. Use run_platform_smoke.sh on macOS/Linux."
}

$scriptDir = Split-Path -Parent $PSCommandPath
$repoRoot = Resolve-Path (Join-Path $scriptDir "../../..")
Set-Location $repoRoot

cargo test `
    --manifest-path crates/openasr-system-audio/Cargo.toml `
    windows_wasapi_system_audio_smoke_emits_non_silent_frames `
    -- `
    --ignored `
    --nocapture
