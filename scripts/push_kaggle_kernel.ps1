<#!
.SYNOPSIS
Pushes and starts the Phase-1 Kaggle kernel without storing or printing credentials.

.DESCRIPTION
Uses the official Kaggle REST endpoint behind `kaggle kernels push`. Credentials
come only from KAGGLE_API_TOKEN, the legacy environment pair, or user-local
~/.kaggle/kaggle.json. The script never writes, prints, or adds credentials to
the repository. It sends a Git archive of tracked source only and attaches the
GeoLifeCLEF 2025 competition as a server-side input.
#>
[CmdletBinding()]
param(
    [string]$KernelSlug = "",
    [ValidateSet("audit", "schema", "frequency", "frequency_smoke", "landsat_smoke", "landsat_scale")]
    [string]$RunMode = "schema"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-KaggleAuthorization {
    if (-not [string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable("KAGGLE_API_TOKEN"))) {
        return [pscustomobject]@{ Scheme = "Bearer"; Value = [Environment]::GetEnvironmentVariable("KAGGLE_API_TOKEN"); Username = $null }
    }
    if ((-not [string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable("KAGGLE_USERNAME"))) -and
        (-not [string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable("KAGGLE_KEY")))) {
        $legacy = "{0}:{1}" -f [Environment]::GetEnvironmentVariable("KAGGLE_USERNAME"), [Environment]::GetEnvironmentVariable("KAGGLE_KEY")
        return [pscustomobject]@{ Scheme = "Basic"; Value = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($legacy)); Username = [Environment]::GetEnvironmentVariable("KAGGLE_USERNAME") }
    }
    # Explicitly supported repository-local credential supplied by the project
    # owner. It is Git-ignored and is read in memory only.
    $repositoryConfigPath = Join-Path $PSScriptRoot "..\api_key\kaggle_2.json"
    if (Test-Path -LiteralPath $repositoryConfigPath) {
        $repositoryConfig = Get-Content -LiteralPath $repositoryConfigPath -Raw | ConvertFrom-Json
        if (-not [string]::IsNullOrWhiteSpace($repositoryConfig.username) -and -not [string]::IsNullOrWhiteSpace($repositoryConfig.key)) {
            $repositoryPair = "{0}:{1}" -f $repositoryConfig.username, $repositoryConfig.key
            return [pscustomobject]@{ Scheme = "Basic"; Value = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($repositoryPair)); Username = $repositoryConfig.username }
        }
        throw "The explicitly configured repository-local Kaggle credential is incomplete."
    }
    $configPath = Join-Path ([Environment]::GetFolderPath("UserProfile")) ".kaggle\kaggle.json"
    if (-not (Test-Path -LiteralPath $configPath)) { throw "No Kaggle authentication is available. Configure KAGGLE_API_TOKEN, legacy environment variables, or ~/.kaggle/kaggle.json." }
    $config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
    if ([string]::IsNullOrWhiteSpace($config.username) -or [string]::IsNullOrWhiteSpace($config.key)) { throw "The user-local Kaggle configuration is incomplete." }
    $pair = "{0}:{1}" -f $config.username, $config.key
    return [pscustomobject]@{ Scheme = "Basic"; Value = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($pair)); Username = $config.username }
}

$auth = Get-KaggleAuthorization
$enableGpu = $RunMode -in @("landsat_smoke", "landsat_scale")
if ([string]::IsNullOrWhiteSpace($KernelSlug)) {
    if ([string]::IsNullOrWhiteSpace($auth.Username)) { throw "When using KAGGLE_API_TOKEN, pass -KernelSlug '<username>/geolifeclef-risk-aware-sdm-phase-1'." }
    $KernelSlug = "$($auth.Username)/geolifeclef-risk-aware-sdm-phase-1"
}

$temporaryArchive = Join-Path ([IO.Path]::GetTempPath()) ("geolifeclef-source-" + [guid]::NewGuid().ToString() + ".zip")
try {
    git -c safe.directory="$PWD" archive --format=zip --output=$temporaryArchive HEAD
    if (-not (Test-Path -LiteralPath $temporaryArchive)) { throw "Could not create the tracked-source archive." }
    $encodedSource = [Convert]::ToBase64String([IO.File]::ReadAllBytes($temporaryArchive))
    $notebookText = @"
import base64
import io
import os
import subprocess
import sys
import zipfile
from pathlib import Path

SOURCE_ARCHIVE_B64 = '$encodedSource'
RUN_MODE = '$RunMode'
PROJECT = Path('/kaggle/working/geolifeclef-risk-aware-sdm')
PROJECT.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(SOURCE_ARCHIVE_B64))) as archive:
    archive.extractall(PROJECT)
os.chdir(PROJECT)

# No credentials are required inside the notebook: the Kaggle API attaches the
# competition source below. --no-deps avoids pulling unrelated packages.
subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--no-build-isolation', '-e', '.'], check=True)

import torch
print({'torch': torch.__version__, 'cuda_available': torch.cuda.is_available(), 'gpu_count': torch.cuda.device_count()})

input_root = Path('/kaggle/input')
competition_roots = [path for path in input_root.rglob('*') if path.is_dir() and path.name == 'geolifeclef-2025']
if not competition_roots:
    raise RuntimeError('GeoLifeCLEF 2025 input was not mounted. Confirm competition rules are accepted, then push again.')
data_root = competition_roots[0]
if RUN_MODE == 'audit':
    subprocess.run([sys.executable, 'scripts/audit_data.py', '--data-root', str(data_root), '--report-dir', 'data/reports'], check=True)
elif RUN_MODE == 'schema':
    subprocess.run([sys.executable, 'scripts/inspect_schema.py', '--data-root', str(data_root), '--report-path', 'data/reports/schema_report.json'], check=True)
elif RUN_MODE == 'frequency':
    subprocess.run([sys.executable, 'scripts/run_frequency_baseline.py', '--metadata-path', str(data_root / 'GLC25_PA_metadata_train.csv'), '--report-path', 'artifacts/frequency_pa/metrics.json'], check=True)
elif RUN_MODE == 'frequency_smoke':
    subprocess.run([sys.executable, 'scripts/run_frequency_baseline.py', '--metadata-path', str(data_root / 'GLC25_PA_metadata_train.csv'), '--report-path', 'artifacts/frequency_pa_smoke/metrics.json', '--max-train-surveys', '12000', '--max-validation-surveys', '3000'], check=True)
elif RUN_MODE == 'landsat_smoke':
    subprocess.run([sys.executable, 'scripts/prepare_landsat_pa.py', '--data-root', str(data_root)], check=True)
    subprocess.run([sys.executable, 'scripts/train.py', '--config', 'configs/landsat_tcn_smoke.yaml'], check=True)
    subprocess.run([sys.executable, 'scripts/evaluate.py', '--checkpoint', 'artifacts/landsat_tcn_smoke/best.pt', '--split', 'data/processed/landsat_pa_smoke_val.npz', '--channels', '32', '--top-k', '16'], check=True)
elif RUN_MODE == 'landsat_scale':
    subprocess.run([sys.executable, 'scripts/prepare_landsat_pa.py', '--data-root', str(data_root), '--train-output', 'data/processed/landsat_pa_scale_train.npz', '--val-output', 'data/processed/landsat_pa_scale_val.npz', '--manifest-path', 'data/processed/landsat_pa_scale_manifest.json', '--max-train-surveys', '48000', '--max-validation-surveys', '6000'], check=True)
    subprocess.run([sys.executable, 'scripts/train.py', '--config', 'configs/landsat_tcn_scale.yaml'], check=True)
    subprocess.run([sys.executable, 'scripts/evaluate.py', '--checkpoint', 'artifacts/landsat_tcn_scale/best.pt', '--split', 'data/processed/landsat_pa_scale_val.npz', '--channels', '32', '--top-k', '16'], check=True)
subprocess.run([sys.executable, '-m', 'pytest'], check=True)

print(f'Phase-1 {RUN_MODE} run and synthetic smoke tests completed.')
print('Training remains deferred until the raw-to-canonical adapter is recorded.')
"@
    $payload = [ordered]@{
        slug = $KernelSlug
        newTitle = "GeoLifeCLEF Risk-Aware SDM - Phase 1"
        text = $notebookText
        language = "python"
        kernelType = "script"
        isPrivate = $true
        # Only neural training requests an accelerator; audit/schema/frequency
        # runs stay CPU-only and therefore do not consume GPU quota.
        enableGpu = $enableGpu
        enableTpu = $false
        enableInternet = $false
        competitionDataSources = @("geolifeclef-2025")
        machineShape = "NvidiaTeslaT4"
    } | ConvertTo-Json -Depth 5 -Compress
    $payloadPath = [IO.Path]::GetTempFileName()
    $responsePath = [IO.Path]::GetTempFileName()
    try {
        [IO.File]::WriteAllText($payloadPath, $payload, [Text.UTF8Encoding]::new($false))
        # curl uses the platform's network configuration. Authorization lives only
        # in this process invocation; it is never written to disk or printed.
        $httpCode = & curl.exe -L --fail --silent --show-error `
            -X POST "https://www.kaggle.com/api/v1/kernels/push" `
            -H "Authorization: $($auth.Scheme) $($auth.Value)" `
            -H "Content-Type: application/json" `
            --data-binary "@$payloadPath" `
            -o $responsePath `
            -w "%{http_code}"
        if ($LASTEXITCODE -ne 0) {
            $serverDetail = "No response detail was returned."
            if ((Test-Path -LiteralPath $responsePath) -and ((Get-Item -LiteralPath $responsePath).Length -gt 0)) {
                $serverDetail = (Get-Content -LiteralPath $responsePath -Raw).Trim()
                # A Kaggle error must never cause an accidental credential echo.
                $serverDetail = [regex]::Replace($serverDetail, '(?i)(token|key|password|authorization)\s*[:=]\s*[^,}\s]+', '$1=[redacted]')
            }
            throw "Kaggle kernel push failed (HTTP $httpCode): $serverDetail"
        }
        $result = Get-Content -LiteralPath $responsePath -Raw | ConvertFrom-Json
        if (-not [string]::IsNullOrWhiteSpace($result.error)) {
            throw "Kaggle rejected the kernel push: $($result.error)"
        }
        if ([int]$result.versionNumber -le 0 -and [string]::IsNullOrWhiteSpace($result.url)) {
            throw "Kaggle did not return a kernel version or URL; no run was confirmed."
        }
    }
    finally {
        if (Test-Path -LiteralPath $payloadPath) { Remove-Item -LiteralPath $payloadPath -Force }
        if (Test-Path -LiteralPath $responsePath) { Remove-Item -LiteralPath $responsePath -Force }
    }
    [pscustomobject]@{ Kernel = $KernelSlug; Version = $result.versionNumber; Ref = $result.ref; Status = "submitted" } | ConvertTo-Json -Compress
}
finally {
    if (Test-Path -LiteralPath $temporaryArchive) { Remove-Item -LiteralPath $temporaryArchive -Force }
}
