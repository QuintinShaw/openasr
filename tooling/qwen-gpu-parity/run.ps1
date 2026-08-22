param()

# qwen GPU correctness producer: it never skips a missing GPU, fixture, pack, or
# trace. The caller must provide the immutable candidate/staging identities and
# the runtime's cold+reuse per-step trace artifacts; transcript equality alone is
# not a correctness receipt.
#
# Run locally on a gfx1200 / CUDA / Vulkan box:
#   cargo build -p openasr-cli --release --features hip   # or cuda / vulkan
#   pwsh tooling/qwen-gpu-parity/run.ps1
#
# Overrides (env):
#   OPENASR_QWEN_PARITY_EXE   path to openasr.exe (default target/release/openasr.exe)
#   OPENASR_QWEN_PARITY_PACK  explicit .oasr pack path (default resolved from OPENASR_HOME)
#   OPENASR_QWEN_PARITY_MODEL model id   (default qwen3-asr-0.6b)
#   OPENASR_QWEN_PARITY_QUANT quant      (default q8_0)
#   OPENASR_QWEN_PARITY_EXPECTED_PROVIDER exact provider selected by the run
#   OPENASR_QWEN_PARITY_EXPECTED_DEVICE exact physical device identity
#   OPENASR_QWEN_PARITY_TRACE_DIR directory containing cold/reuse per-step traces

Set-StrictMode -Version Latest
# The native ggml engine prints an init banner to stderr; under Windows
# PowerShell 5.1 with ErrorActionPreference=Stop that turns into a terminating
# NativeCommandError. Use Continue and gate strictly on $LASTEXITCODE instead.
$ErrorActionPreference = "Continue"
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Exe = if ($env:OPENASR_QWEN_PARITY_EXE) { $env:OPENASR_QWEN_PARITY_EXE } else { Join-Path $Root "target\release\openasr.exe" }
$ModelId = if ($env:OPENASR_QWEN_PARITY_MODEL) { $env:OPENASR_QWEN_PARITY_MODEL } else { "qwen3-asr-0.6b" }
$Quant = if ($env:OPENASR_QWEN_PARITY_QUANT) { $env:OPENASR_QWEN_PARITY_QUANT } else { "q8_0" }
$OpenAsrHome = if ($env:OPENASR_HOME) { $env:OPENASR_HOME } else { Join-Path $env:USERPROFILE ".openasr" }
$ExpectedProvider = if ($env:OPENASR_QWEN_PARITY_EXPECTED_PROVIDER) { $env:OPENASR_QWEN_PARITY_EXPECTED_PROVIDER } else { Fail "OPENASR_QWEN_PARITY_EXPECTED_PROVIDER is required" 2 }
$ExpectedDevice = if ($env:OPENASR_QWEN_PARITY_EXPECTED_DEVICE) { $env:OPENASR_QWEN_PARITY_EXPECTED_DEVICE } else { Fail "OPENASR_QWEN_PARITY_EXPECTED_DEVICE is required" 2 }
$TraceDir = if ($env:OPENASR_QWEN_PARITY_TRACE_DIR) { $env:OPENASR_QWEN_PARITY_TRACE_DIR } else { Fail "OPENASR_QWEN_PARITY_TRACE_DIR is required" 2 }
if (!(Test-Path -LiteralPath $TraceDir)) { Fail "Missing runtime trace directory: $TraceDir" 2 }

if ($env:OPENASR_QWEN_PARITY_AUDIO) {
    $AudioList = @($env:OPENASR_QWEN_PARITY_AUDIO.Split(";") | Where-Object { $_.Trim().Length -gt 0 })
} else {
    $AudioList = @(
        (Join-Path $Root "fixtures\jfk.wav")
    )
}

function Fail {
    param([string]$Message, [int]$Code)
    [Console]::Error.WriteLine($Message)
    exit $Code
}

if (!(Test-Path -LiteralPath $Exe)) {
    Fail "Missing openasr exe: $Exe`nBuild it first, e.g.: cargo build -p openasr-cli --release --features hip" 2
}
if (!(Test-Path -LiteralPath $Pack)) {
    Fail "Missing model pack: $Pack`nPull it first, e.g.: openasr pull $ModelId" 2
}

function Invoke-Transcribe {
    param([string]$Audio, [string]$Backend)
    $prev = [Environment]::GetEnvironmentVariable("OPENASR_GGML_BACKEND", "Process")
    try {
        [Environment]::SetEnvironmentVariable("OPENASR_GGML_BACKEND", $Backend, "Process")
        $out = & $Exe transcribe $Audio --backend native --model-pack $Pack --format text 2>$null
        if ($LASTEXITCODE -ne 0) { return $null }
        return ($out | Out-String).Trim()
    } finally {
        [Environment]::SetEnvironmentVariable("OPENASR_GGML_BACKEND", $prev, "Process")
    }
}

$doctor = & $Exe doctor 2>$null | Out-String
$bestBackendLine = (($doctor -split "`n") | Where-Object { $_ -match "best backend" }) -join " "
Write-Host "exe=$Exe"
Write-Host "pack=$Pack"
Write-Host ("doctor: " + $bestBackendLine.Trim())
if ($bestBackendLine -notmatch [regex]::Escape($ExpectedProvider)) {
    Fail "Expected provider '$ExpectedProvider' was not selected: $bestBackendLine" 1
}
if ($bestBackendLine -notmatch [regex]::Escape($ExpectedDevice)) {
    Fail "Expected physical device '$ExpectedDevice' was not selected: $bestBackendLine" 1
}
$traceFiles = @(Get-ChildItem -LiteralPath $TraceDir -File -Filter "*.jsonl")
if (!($traceFiles.Name -match "cold") -or !($traceFiles.Name -match "reuse")) {
    Fail "Trace directory must contain explicitly named cold and reuse traces" 1
}

$failures = 0
foreach ($audio in $AudioList) {
    if (!(Test-Path -LiteralPath $audio)) {
        Fail "Missing required fixture: $audio" 1
    }
    $name = Split-Path -Leaf $audio
    $cpu = Invoke-Transcribe -Audio $audio -Backend "cpu"
    if ($null -eq $cpu) { Write-Warning "CPU transcribe FAILED for $name"; $failures += 1; continue }
    $gpu = Invoke-Transcribe -Audio $audio -Backend ""  # empty => auto-select (GPU)
    if ($null -eq $gpu) { Write-Warning "default(GPU) transcribe FAILED for $name"; $failures += 1; continue }
    if ($cpu -eq $gpu) {
        Write-Host "PASS  $name  GPU==CPU : $gpu"
    } else {
        Write-Host "FAIL  $name  GPU!=CPU"
        Write-Host "  CPU: $cpu"
        Write-Host "  GPU: $gpu"
        $failures += 1
    }
}

if ($failures -ne 0) {
    Fail "qwen GPU parity gate: $failures mismatch/failure(s) - qwen GPU output diverges from the CPU reference." 1
}
Write-Host "qwen GPU parity gate: PASS (GPU transcript matches CPU reference for all fixtures)."
