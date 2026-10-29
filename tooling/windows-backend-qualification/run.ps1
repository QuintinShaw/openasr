param()

# Exact-cell producer for an already prepared, isolated backend qualification
# scope. This script creates the existing real-family receipts and trace
# schema; gpu_correctness_gate.py remains the sole binding/approval authority.

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $true
}

function Fail {
    param([string]$Message)
    throw $Message
}

function Required-Env {
    param([string]$Name)
    $value = [Environment]::GetEnvironmentVariable($Name, "Process")
    if ([string]::IsNullOrWhiteSpace($value)) { Fail "$Name is required" }
    return $value
}

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$PythonCommand = Get-Command python3 -ErrorAction SilentlyContinue
if (-not $PythonCommand) { $PythonCommand = Get-Command python -ErrorAction SilentlyContinue }
if (-not $PythonCommand) { Fail "python3 or python is required" }
$Python = $PythonCommand.Source
$Exe = Required-Env "OPENASR_QUALIFICATION_EXE"
$MatrixPath = Required-Env "OPENASR_QUALIFICATION_MATRIX"
$InventoryPath = Required-Env "OPENASR_QUALIFICATION_INVENTORY"
$ModelCatalogPath = Required-Env "OPENASR_QUALIFICATION_MODEL_CATALOG"
$BackendCatalogPath = Required-Env "OPENASR_QUALIFICATION_BACKEND_CATALOG"
$BackendId = Required-Env "OPENASR_QUALIFICATION_BACKEND_ID"
$ModelId = Required-Env "OPENASR_QUALIFICATION_MODEL_ID"
$Quant = Required-Env "OPENASR_QUALIFICATION_QUANT"
$PluginPath = Required-Env "OPENASR_QUALIFICATION_PLUGIN"
$PackPath = Required-Env "OPENASR_QUALIFICATION_PACK"
$AudioPath = Required-Env "OPENASR_QUALIFICATION_AUDIO"
$OutputDir = Required-Env "OPENASR_QUALIFICATION_OUTPUT_DIR"
$QualificationScope = Required-Env "OPENASR_BACKEND_QUALIFICATION_SCOPE"

foreach ($path in @($Exe, $MatrixPath, $InventoryPath, $ModelCatalogPath, $BackendCatalogPath, $PluginPath, $PackPath, $AudioPath)) {
    if (!(Test-Path -LiteralPath $path -PathType Leaf)) { Fail "required file is missing: $path" }
}
if ($QualificationScope -notmatch '^[A-Za-z0-9][A-Za-z0-9_.:+@=-]{0,255}/[0-9a-f]{32}$') {
    Fail "qualification scope is not one privacy-safe runner scope"
}

$matrix = Get-Content -LiteralPath $MatrixPath -Raw | ConvertFrom-Json
$cells = @($matrix.cells | Where-Object {
    $_.backend_id -eq $BackendId -and $_.model_id -eq $ModelId -and $_.quant -eq $Quant
})
if ($cells.Count -ne 1) { Fail "matrix must contain exactly one requested backend/model/quant cell" }
$cell = $cells[0]
$Provider = [string]$cell.provider
$CoreCommit = [string]$matrix.artifact_contract.core_commit
if ($Provider -notin @("cuda", "hip", "vulkan")) { Fail "matrix cell has an unsupported provider" }
if ($CoreCommit -notmatch '^[0-9a-f]{40}$') { Fail "matrix has an invalid core commit" }
$cellBytes = [Text.Encoding]::UTF8.GetBytes("$BackendId`0$ModelId`0$Quant")
$cellDigest = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($cellBytes)).ToLowerInvariant()
$cellKey = $cellDigest.Substring(0, 16)

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$existing = @(Get-ChildItem -LiteralPath $OutputDir -File | Where-Object {
    $_.Name -like "gpu-correctness-receipt-*" -or $_.Name -like "gpu-correctness-trace-*"
})
if ($existing.Count -ne 0) { Fail "refuse to overwrite existing qualification evidence" }
$scratch = Join-Path $OutputDir ("raw-" + [Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $scratch | Out-Null

$previousBackend = [Environment]::GetEnvironmentVariable("OPENASR_GGML_BACKEND", "Process")
$previousOffline = [Environment]::GetEnvironmentVariable("OPENASR_OFFLINE", "Process")
try {
    [Environment]::SetEnvironmentVariable("OPENASR_OFFLINE", "1", "Process")
    [Environment]::SetEnvironmentVariable("OPENASR_GGML_BACKEND", $Provider, "Process")
    # Discover the stable device through an actual admitted request. Caller
    # labels never substitute for the runtime's selected provider/device.
    $discoveryPath = Join-Path $scratch "discovery.json"
    & $Exe bench-receipt short-audio --model "$ModelId`:$Quant" `
        --model-pack $PackPath --audio $AudioPath --backend native --device $Provider `
        --out $discoveryPath --runs 1 --warmup-runs 0 --core-commit $CoreCommit
    if ($LASTEXITCODE -ne 0) { Fail "GPU device discovery request failed" }
    $discovery = Get-Content -LiteralPath $discoveryPath -Raw | ConvertFrom-Json
    if ($discovery.actual_provider -ne $Provider -or [string]::IsNullOrWhiteSpace($discovery.actual_stable_device_id)) {
        Fail "GPU discovery did not execute on the requested provider"
    }
    $scopeBytes = [Text.Encoding]::UTF8.GetBytes($QualificationScope)
    $scopeDigest = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($scopeBytes)).ToLowerInvariant()
    $recordPath = Join-Path (Required-Env "OPENASR_HOME") "backends/qualification/$scopeDigest.json"
    $record = Get-Content -LiteralPath $recordPath -Raw | ConvertFrom-Json
    if ($record.backend_id -ne $BackendId -or $record.vendor -ne $Provider -or
        $record.device_target -ne $cell.device_target -or $record.artifact_fingerprint -ne $cell.artifact_fingerprint) {
        Fail "prepared qualification record does not match the exact cell"
    }
    function Digest([string]$Path) { return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() }
    function Artifact([string]$Path) { return @{ label = [IO.Path]::GetFileName($Path); sha256 = (Digest $Path); size_bytes = (Get-Item -LiteralPath $Path).Length } }
    $binding = @{
        matrix_sha256 = $matrix.matrix_sha256
        candidate_release_subject = $matrix.artifact_contract.release_subject
        catalog_digests = @{ inventory_sha256 = (Digest $InventoryPath); model_catalog_sha256 = (Digest $ModelCatalogPath); backend_catalog_sha256 = (Digest $BackendCatalogPath) }
        family = $cell.family; model_id = $ModelId; quant = $Quant
        topology = $cell.topology.decoder_state; provider = $Provider; device_target = $cell.device_target
        backend_id = $BackendId; driver_version = $record.driver_version; artifact_fingerprint = $cell.artifact_fingerprint
        device = $discovery.actual_stable_device_id; placement = $cell.placement
        capture_mode = $cell.capture_mode; scheduler_mode = $cell.scheduler_mode; graph_mode = $cell.graph_mode
        output_plan = $cell.output_plan
        artifacts = @{ binary = (Artifact $Exe); plugin = (Artifact $PluginPath); pack = (Artifact $PackPath); fixture = (Artifact $AudioPath) }
    }
    $bindingPath = Join-Path $scratch "binding.json"
    $binding | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $bindingPath -Encoding utf8NoBOM
    # One invocation owns both requests, so cold/reuse share native services,
    # process identity and graph caches. Full logits are emitted by this route.
    & $Exe bench-receipt qualify-family --model "$ModelId`:$Quant" `
        --model-pack $PackPath --audio $AudioPath --device $Provider `
        --binding $bindingPath --out-dir $scratch --core-commit $CoreCommit
    if ($LASTEXITCODE -ne 0) { Fail "same-process real-family producer failed" }
    foreach ($Mode in @("cold", "reuse")) {
        $warmups = if ($Mode -eq "reuse") { 1 } else { 0 }
        $nonce = [Guid]::NewGuid().ToString("N").ToLowerInvariant()
        $scope = "gpu-correctness/$nonce"
        $cpuReceipt = Join-Path $scratch "cpu-$Mode.json"
        $cpuTrace = Join-Path $scratch "cpu-$Mode.jsonl"
        $cpuLogits = Join-Path $scratch "cpu-$Mode-logits.jsonl"
        $gpuReceipt = Join-Path $scratch "diagnostic-$Mode.json"
        $traceName = "gpu-correctness-trace-$Provider-$cellKey-$Mode.jsonl"
        $gpuTrace = Join-Path $OutputDir $traceName
        $gpuLogits = Join-Path $OutputDir "gpu-correctness-trace-$Provider-$cellKey-$Mode-logits.jsonl"
        [IO.File]::Copy((Join-Path $scratch "token-$Mode.jsonl"), $gpuTrace, $false)
        [IO.File]::Copy((Join-Path $scratch "logits-$Mode.jsonl"), $gpuLogits, $false)

        [Environment]::SetEnvironmentVariable("OPENASR_GGML_BACKEND", "cpu", "Process")
        & $Exe bench-receipt short-audio --model "$ModelId`:$Quant" `
            --model-pack $PackPath --audio $AudioPath --backend native --device cpu `
            --out $cpuReceipt --trace-out $cpuTrace --logits-out $cpuLogits --runs 1 --warmup-runs $warmups `
            --scope $scope --core-commit $CoreCommit
        if ($LASTEXITCODE -ne 0) { Fail "CPU oracle receipt failed for $Mode" }

        $placementOut = Join-Path $OutputDir "gpu-correctness-receipt-$Provider-$cellKey-$Mode-placement.json"
        $tokenOut = Join-Path $OutputDir "gpu-correctness-receipt-$Provider-$cellKey-$Mode-token.json"
        & $Python (Join-Path $Root "tooling\release-manifest\gpu_correctness_gate.py") bind-cell `
            --manifest $MatrixPath --inventory $InventoryPath --catalog $ModelCatalogPath `
            --backend-catalog $BackendCatalogPath --backend-id $BackendId --process-mode $Mode `
            --gpu-receipt $gpuReceipt --gpu-trace $gpuTrace --gpu-logits $gpuLogits `
            --cpu-receipt $cpuReceipt --cpu-trace $cpuTrace `
            --binary $Exe --plugin $PluginPath --pack $PackPath --fixture $AudioPath `
            --placement-out $placementOut --token-out $tokenOut
        if ($LASTEXITCODE -ne 0) { Fail "correctness binding failed for $Mode" }
    }

    $receipts = @(Get-ChildItem -LiteralPath $OutputDir -Filter "gpu-correctness-receipt-*.json" -File)
    $traces = @(Get-ChildItem -LiteralPath $OutputDir -Filter "gpu-correctness-trace-*.jsonl" -File)
    if ($receipts.Count -ne 4 -or $traces.Count -ne 4) {
        Fail "qualification output must contain four receipts and four traces"
    }
    $validateArgs = @(
        (Join-Path $Root "tooling\release-manifest\gpu_correctness_gate.py"), "validate",
        "--manifest", $MatrixPath, "--inventory", $InventoryPath,
        "--catalog", $ModelCatalogPath, "--backend-catalog", $BackendCatalogPath
    )
    foreach ($receipt in $receipts) { $validateArgs += @("--receipt", $receipt.FullName) }
    foreach ($trace in $traces) { $validateArgs += @("--trace", $trace.FullName) }
    & $Python @validateArgs
    if ($LASTEXITCODE -ne 0) { Fail "bound exact-cell evidence failed final validation" }
} finally {
    [Environment]::SetEnvironmentVariable("OPENASR_GGML_BACKEND", $previousBackend, "Process")
    [Environment]::SetEnvironmentVariable("OPENASR_OFFLINE", $previousOffline, "Process")
    Write-Host "Raw qualification inputs and failure evidence retained: $scratch"
}

Write-Host "QUALIFICATION-CELL-PASSED provider=$Provider backend_id=$BackendId model=$ModelId quant=$Quant"
