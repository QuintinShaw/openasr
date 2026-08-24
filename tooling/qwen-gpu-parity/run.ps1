param()

# qwen GPU raw diagnostic. It never skips a missing GPU, fixture, pack, or
# trace, but caller expectations and local activation state are not release
# authority. Output names deliberately cannot match the release gate globs.
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
function Fail {
    param([string]$Message, [int]$Code)
    [Console]::Error.WriteLine($Message)
    exit $Code
}

$Exe = if ($env:OPENASR_QWEN_PARITY_EXE) { $env:OPENASR_QWEN_PARITY_EXE } else { Join-Path $Root "target\release\openasr.exe" }
$ModelId = if ($env:OPENASR_QWEN_PARITY_MODEL) { $env:OPENASR_QWEN_PARITY_MODEL } else { "qwen3-asr-0.6b" }
$Quant = if ($env:OPENASR_QWEN_PARITY_QUANT) { $env:OPENASR_QWEN_PARITY_QUANT } else { "q8_0" }
$OpenAsrHome = if ($env:OPENASR_HOME) { $env:OPENASR_HOME } else { Join-Path $env:USERPROFILE ".openasr" }
$Pack = if ($env:OPENASR_QWEN_PARITY_PACK) { $env:OPENASR_QWEN_PARITY_PACK } else { Join-Path $OpenAsrHome ("models\{0}\{1}\{0}-{1}.oasr" -f $ModelId, $Quant) }
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

function Invoke-GpuDiagnostic {
    param([string]$Audio, [ValidateSet("cold", "reuse")][string]$Mode)
    $name = Split-Path -Leaf $Audio
    $diagnosticPath = Join-Path $TraceDir ("qwen-{0}-{1}.diagnostic.json" -f $Mode, $name)
    $tracePath = Join-Path $TraceDir ("qwen-{0}-{1}.diagnostic.jsonl" -f $Mode, $name)
    if (Test-Path $diagnosticPath) { Remove-Item $diagnosticPath -Force }
    if (Test-Path $tracePath) { Remove-Item $tracePath -Force }
    $warmups = if ($Mode -eq "reuse") { 1 } else { 0 }
    $scope = "qwen-gpu-parity/" + ([Guid]::NewGuid().ToString("N").ToLowerInvariant())
    & $Exe bench-receipt short-audio --model ("{0}:{1}" -f $ModelId, $Quant) `
        --model-pack $Pack --audio $Audio --backend native --device $ExpectedProvider `
        --out $diagnosticPath --trace-out $tracePath --runs 1 --warmup-runs $warmups `
        --scope $scope 2>$null
    if ($LASTEXITCODE -ne 0 -or !(Test-Path $diagnosticPath) -or !(Test-Path $tracePath)) {
        return $null
    }
    $diagnostic = Get-Content -LiteralPath $diagnosticPath -Raw | ConvertFrom-Json
    $header = Get-Content -LiteralPath $tracePath -TotalCount 1 | ConvertFrom-Json
    if ($header.provider -ne $ExpectedProvider -or $header.device -ne $ExpectedDevice -or `
        $header.graph_mode -notin @("fresh_graph", "reusable_graph")) {
        return $null
    }
    return $diagnostic.transcript.text
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
$failures = 0
foreach ($audio in $AudioList) {
    if (!(Test-Path -LiteralPath $audio)) {
        Fail "Missing required fixture: $audio" 1
    }
    $name = Split-Path -Leaf $audio
    $cpu = Invoke-Transcribe -Audio $audio -Backend "cpu"
    if ($null -eq $cpu) { Write-Warning "CPU transcribe FAILED for $name"; $failures += 1; continue }
    $gpuCold = Invoke-GpuDiagnostic -Audio $audio -Mode "cold"
    $gpuReuse = Invoke-GpuDiagnostic -Audio $audio -Mode "reuse"
    if ($null -eq $gpuCold -or $null -eq $gpuReuse) { Write-Warning "GPU cold/reuse transcribe FAILED for $name"; $failures += 1; continue }
    if ($cpu -eq $gpuCold -and $gpuCold -eq $gpuReuse) {
        Write-Host "PASS  $name  GPU cold/reuse==CPU : $gpuCold"
    } else {
        Write-Host "FAIL  $name  GPU!=CPU"
        Write-Host "  CPU: $cpu"
        Write-Host "  GPU cold: $gpuCold"
        Write-Host "  GPU reuse: $gpuReuse"
        $failures += 1
    }
}

$traceFiles = @(Get-ChildItem -LiteralPath $TraceDir -File -Filter "*.diagnostic.jsonl")
if ($traceFiles.Count -lt 2 -or !($traceFiles.Name -match "cold") -or !($traceFiles.Name -match "reuse")) {
    Fail "Runtime did not emit both cold and reuse per-step traces" 1
}

if ($failures -ne 0) {
    Fail "qwen GPU parity diagnostic: $failures mismatch/failure(s) - qwen GPU output diverges from the CPU reference." 1
}
Write-Host "qwen GPU parity diagnostic: MATCH (not release or activation evidence)."
