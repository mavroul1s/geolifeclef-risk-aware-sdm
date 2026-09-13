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
    [ValidateSet("audit", "schema", "spatial_audit", "frequency", "frequency_smoke", "landsat_smoke", "landsat_scale", "landsat_full", "sota_spatial_smoke", "sota_spatial_full", "sota_spatial_multiseed", "sota_spatial_multiseed_resume", "official_pa_submit", "sota_single", "environmental_challenger", "ood_po_expert_v21")]
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
$enableGpu = $RunMode -in @("landsat_smoke", "landsat_scale", "landsat_full", "sota_spatial_smoke", "sota_spatial_full", "sota_spatial_multiseed", "sota_spatial_multiseed_resume", "official_pa_submit", "sota_single", "environmental_challenger", "ood_po_expert_v21")
if ([string]::IsNullOrWhiteSpace($KernelSlug)) {
    if ([string]::IsNullOrWhiteSpace($auth.Username)) { throw "When using KAGGLE_API_TOKEN, pass -KernelSlug '<username>/geolifeclef-risk-aware-sdm-phase-1'." }
    $KernelSlug = "$($auth.Username)/geolifeclef-risk-aware-sdm-phase-1"
}

$temporaryArchive = Join-Path ([IO.Path]::GetTempPath()) ("geolifeclef-source-" + [guid]::NewGuid().ToString() + ".zip")
try {
    git -c safe.directory="$PWD" archive --format=zip --output=$temporaryArchive HEAD
    if (-not (Test-Path -LiteralPath $temporaryArchive)) { throw "Could not create the tracked-source archive." }
    $encodedSource = [Convert]::ToBase64String([IO.File]::ReadAllBytes($temporaryArchive))
    $sourceCommit = git rev-parse HEAD
    $notebookText = @"
import base64
import io
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

SOURCE_ARCHIVE_B64 = '$encodedSource'
RUN_MODE = '$RunMode'
os.environ['GLC_PIPELINE_STARTED_AT'] = str(time.time())
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['GLC_SOURCE_COMMIT'] = '$sourceCommit'
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
competition_roots = [path for path in (input_root / 'competitions' / 'geolifeclef-2025', input_root / 'geolifeclef-2025') if path.is_dir()]
if not competition_roots:
    raise RuntimeError('GeoLifeCLEF 2025 input was not mounted. Confirm competition rules are accepted, then push again.')
data_root = competition_roots[0]
if RUN_MODE == 'audit':
    subprocess.run([sys.executable, 'scripts/audit_data.py', '--data-root', str(data_root), '--report-dir', 'data/reports'], check=True)
elif RUN_MODE == 'schema':
    subprocess.run([sys.executable, 'scripts/inspect_schema.py', '--data-root', str(data_root), '--report-path', 'data/reports/schema_report.json'], check=True)
elif RUN_MODE == 'spatial_audit':
    subprocess.run([sys.executable, 'scripts/audit_spatial_split.py', '--metadata-path', str(data_root / 'GLC25_PA_metadata_train.csv'), '--report-path', 'data/reports/spatial_split_audit.json'], check=True)
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
elif RUN_MODE == 'landsat_full':
    subprocess.run([sys.executable, 'scripts/prepare_landsat_pa.py', '--data-root', str(data_root), '--train-output', 'data/processed/landsat_pa_full_train.npz', '--val-output', 'data/processed/landsat_pa_full_val.npz', '--manifest-path', 'data/processed/landsat_pa_full_manifest.json', '--max-train-surveys', '71190', '--max-validation-surveys', '17797'], check=True)
    subprocess.run([sys.executable, 'scripts/train.py', '--config', 'configs/landsat_tcn_full.yaml'], check=True)
    subprocess.run([sys.executable, 'scripts/evaluate.py', '--checkpoint', 'artifacts/landsat_tcn_full/best.pt', '--split', 'data/processed/landsat_pa_full_val.npz', '--channels', '32', '--top-k', '16'], check=True)
elif RUN_MODE == 'sota_spatial_smoke':
    subprocess.run([sys.executable, '-m', 'pytest'], check=True)
    subprocess.run([sys.executable, 'scripts/prepare_spatial_multimodal.py', '--data-root', str(data_root), '--output-dir', 'data/processed/sota_spatial_smoke', '--holdout-country', 'Netherlands', '--image-size', '32', '--max-train-surveys', '12000', '--max-calibration-surveys', '2000', '--max-validation-surveys', '3000'], check=True)
    subprocess.run([sys.executable, 'scripts/train_spatial_competition.py', '--data-dir', 'data/processed/sota_spatial_smoke', '--output-dir', 'artifacts/sota_spatial_smoke', '--batch-size', '96', '--reference-epochs', '3', '--fusion-epochs', '12', '--model-dim', '96', '--max-hours', '3.0', '--cleanup-cache'], check=True)
elif RUN_MODE == 'sota_spatial_full':
    subprocess.run([sys.executable, '-m', 'pytest'], check=True)
    subprocess.run([sys.executable, 'scripts/prepare_spatial_multimodal.py', '--data-root', str(data_root), '--output-dir', 'data/processed/sota_spatial_full', '--holdout-country', 'Netherlands', '--image-size', '32'], check=True)
    subprocess.run([sys.executable, 'scripts/train_spatial_competition.py', '--data-dir', 'data/processed/sota_spatial_full', '--output-dir', 'artifacts/sota_spatial_full', '--batch-size', '64', '--reference-epochs', '8', '--fusion-epochs', '16', '--model-dim', '192', '--max-hours', '10.5', '--cleanup-cache'], check=True)
elif RUN_MODE == 'sota_spatial_multiseed':
    subprocess.run([sys.executable, '-m', 'pytest'], check=True)
    data_dir = 'data/processed/sota_spatial_multiseed'
    subprocess.run([sys.executable, 'scripts/prepare_spatial_multimodal.py', '--data-root', str(data_root), '--output-dir', data_dir, '--holdout-country', 'Netherlands', '--image-size', '32'], check=True)
    run_dirs = []
    for seed in ('2025', '3407', '7919'):
        run_dir = f'artifacts/sota_spatial_multiseed_seed_{seed}'
        run_dirs.append(run_dir)
        subprocess.run([sys.executable, 'scripts/train_spatial_competition.py', '--data-dir', data_dir, '--output-dir', run_dir, '--seed', seed, '--batch-size', '64', '--reference-epochs', '8', '--fusion-epochs', '16', '--model-dim', '192', '--max-hours', '3.0'], check=True)
    subprocess.run([sys.executable, 'scripts/evaluate_spatial_ensemble.py', '--data-dir', data_dir, '--run-dirs', *run_dirs, '--output-path', 'artifacts/sota_spatial_multiseed/ensemble.json', '--model-dim', '192', '--batch-size', '64', '--cleanup-cache'], check=True)
elif RUN_MODE == 'sota_spatial_multiseed_resume':
    subprocess.run([sys.executable, '-m', 'pytest'], check=True)
    comparison_files = sorted(input_root.rglob('artifacts/sota_spatial_multiseed_seed_2025/comparison.json'))
    if len(comparison_files) != 1:
        raise RuntimeError(f'Expected one mounted version-16 output, found {len(comparison_files)} candidates.')
    previous_project = comparison_files[0].parents[2]
    previous_data = previous_project / 'data/processed/sota_spatial_multiseed'
    previous_runs = [previous_project / f'artifacts/sota_spatial_multiseed_seed_{seed}' for seed in ('2025', '3407', '7919')]
    subprocess.run([sys.executable, 'scripts/evaluate_spatial_ensemble.py', '--data-dir', str(previous_data), '--run-dirs', *map(str, previous_runs), '--output-path', 'artifacts/sota_spatial_multiseed/ensemble.json', '--model-dim', '192', '--batch-size', '64'], check=True)
elif RUN_MODE == 'official_pa_submit':
    subprocess.run([sys.executable, '-m', 'pytest'], check=True)
    data_dir = 'data/processed/official_pa'
    output_dir = 'artifacts/official_pa_submission'
    subprocess.run([sys.executable, 'scripts/prepare_official_pa.py', '--data-root', str(data_root), '--output-dir', data_dir, '--image-size', '32'], check=True)
    subprocess.run([sys.executable, 'scripts/train_full_pa_ensemble.py', '--data-dir', data_dir, '--sample-submission', str(data_root / 'GLC25_SAMPLE_SUBMISSION.csv'), '--output-dir', output_dir, '--seeds', '2025', '3407', '7919', '--epochs', '16', '--model-dim', '192', '--batch-size', '64', '--top-k', '18', '--max-hours', '10.5', '--cleanup-cache'], check=True)
elif RUN_MODE == 'sota_single':
    subprocess.run([sys.executable, '-m', 'pytest'], check=True)
    data_dir = 'data/processed/sota_single'
    output_dir = 'artifacts/sota_single'
    subprocess.run([sys.executable, 'scripts/prepare_official_pa.py', '--data-root', str(data_root), '--output-dir', data_dir, '--image-size', '32'], check=True)
    subprocess.run([sys.executable, 'scripts/train_sota_single.py', '--data-root', str(data_root), '--data-dir', data_dir, '--sample-submission', str(data_root / 'GLC25_SAMPLE_SUBMISSION.csv'), '--output-dir', output_dir, '--seed', '2025', '--stage-one-epochs', '12', '--full-finetune-epochs', '6', '--model-dim', '192', '--batch-size', '64', '--rare-max-occurrences', '50', '--max-hours', '10.5', '--cleanup-cache'], check=True)
elif RUN_MODE == 'environmental_challenger':
    subprocess.run([sys.executable, '-m', 'pytest'], check=True)
    remaining_seconds = max(1, 10.5 * 3600 - (time.time() - float(os.environ['GLC_PIPELINE_STARTED_AT'])))
    subprocess.run([sys.executable, '-m', 'scripts.run_environmental_challenger', '--data-root', str(data_root), '--epochs', '24', '--batch-size', '128', '--max-hours', '10.5'], check=True, timeout=remaining_seconds)
elif RUN_MODE == 'ood_po_expert_v21':
    import json
    master = json.loads(Path('notebooks/geolifeclef_research_pipeline.ipynb').read_text())
    for cell in master['cells']:
        if cell['cell_type'] == 'code':
            exec(compile(''.join(cell['source']), 'master_notebook', 'exec'))
if RUN_MODE != 'ood_po_expert_v21':
    subprocess.run([sys.executable, '-m', 'pytest'], check=True)

print(f'Phase-1 {RUN_MODE} run and synthetic smoke tests completed.')
"@
    $kernelSources = @()
    $datasetSources = @()
    if ($RunMode -eq "ood_po_expert_v21") {
        $datasetSources = @("con1los/geolifeclef-v20-frozen-control/1")
    }
    $kernelTitle = "GeoLifeCLEF Risk-Aware SDM - Phase 1"
    $kernelType = "script"
    if ($RunMode -in @("environmental_challenger", "ood_po_expert_v21")) {
        $kernelType = "notebook"
        $notebookDocument = [ordered]@{
            cells = @(
                [ordered]@{ cell_type = "markdown"; metadata = @{}; source = @("# GeoLifeCLEF 2025: $RunMode`n", "One competition-only bounded run. See the embedded master notebook and committed preregistration. No SOTA claim before official evaluation.") },
                [ordered]@{ cell_type = "code"; execution_count = $null; metadata = @{}; outputs = @(); source = @($notebookText) }
            )
            metadata = @{ kernelspec = @{ display_name = "Python 3"; language = "python"; name = "python3" }; language_info = @{ name = "python"; version = "3.12" } }
            nbformat = 4
            nbformat_minor = 5
        }
        $notebookText = $notebookDocument | ConvertTo-Json -Depth 12 -Compress
    }
    if ($RunMode -eq "sota_spatial_multiseed_resume") {
        if ([string]::IsNullOrWhiteSpace($auth.Username)) { throw "The resume mode requires a username-based Kaggle credential." }
        $kernelSources = @("$($auth.Username)/geolifeclef-risk-aware-sdm-phase-1")
        $kernelTitle = "GeoLifeCLEF Risk-Aware SDM Ensemble Recovery"
    }
    $payload = [ordered]@{
        slug = $KernelSlug
        newTitle = $kernelTitle
        text = $notebookText
        language = "python"
        kernelType = $kernelType
        isPrivate = $true
        # Only neural training requests an accelerator; audit/schema/frequency
        # runs stay CPU-only and therefore do not consume GPU quota.
        enableGpu = $enableGpu
        enableTpu = $false
        enableInternet = $false
        competitionDataSources = @("geolifeclef-2025")
        kernelDataSources = $kernelSources
        datasetDataSources = $datasetSources
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
