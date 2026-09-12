<#!
.SYNOPSIS
Validates and submits the official PA CSV produced by a completed Kaggle kernel.

.DESCRIPTION
Downloads only the submission CSV and its manifest to temporary files, verifies
their registered invariants and SHA-256 digest, then submits once through the
official Kaggle API client. Credentials and signed output URLs are never printed.
#>
[CmdletBinding()]
param(
    [string]$KernelSlug = "con1los/geolifeclef-risk-aware-sdm-phase-1",
    [string]$Competition = "geolifeclef-2025",
    [string]$Message = "CompetitiveFusionSDM full-PA 3-seed top-18 ensemble",
    [string]$ArtifactDirectory = "official_pa_submission",
    [string]$ManifestName = "submission_manifest.json",
    [switch]$AdaptivePolicy
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$credentialPath = Join-Path $PSScriptRoot "..\api_key\kaggle_2.json"
if (-not (Test-Path -LiteralPath $credentialPath)) {
    throw "The explicitly authorized repository-local Kaggle credential was not found."
}
$credential = Get-Content -LiteralPath $credentialPath -Raw | ConvertFrom-Json
if ([string]::IsNullOrWhiteSpace($credential.username) -or [string]::IsNullOrWhiteSpace($credential.key)) {
    throw "The repository-local Kaggle credential is incomplete."
}
$pair = "{0}:{1}" -f $credential.username, $credential.key
$auth = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($pair))
$slugParts = $KernelSlug.Split("/", 2)
if ($slugParts.Count -ne 2) { throw "KernelSlug must be '<owner>/<slug>'." }

$statusJson = & curl.exe -L --fail --silent --show-error -G `
    -H "Authorization: Basic $auth" `
    --data-urlencode "userName=$($slugParts[0])" `
    --data-urlencode "kernelSlug=$($slugParts[1])" `
    "https://www.kaggle.com/api/v1/kernels/status"
if ($LASTEXITCODE -ne 0) { throw "Could not read the Kaggle kernel status." }
$status = $statusJson | ConvertFrom-Json
if ($status.status -ne "complete") {
    throw "Kaggle kernel must be complete before submission; current status is '$($status.status)'."
}

$metadataFile = New-TemporaryFile
$submissionFile = Join-Path ([IO.Path]::GetTempPath()) ("GLC25_PA_submission_" + [guid]::NewGuid().ToString() + ".csv")
$manifestFile = New-TemporaryFile
$templateFile = New-TemporaryFile
try {
    & curl.exe -L --fail --silent --show-error -G `
        -H "Authorization: Basic $auth" `
        --data-urlencode "userName=$($slugParts[0])" `
        --data-urlencode "kernelSlug=$($slugParts[1])" `
        "https://www.kaggle.com/api/v1/kernels/output" `
        -o $metadataFile.FullName
    if ($LASTEXITCODE -ne 0) { throw "Could not list completed kernel outputs." }
    $outputs = Get-Content -LiteralPath $metadataFile.FullName -Raw | ConvertFrom-Json
    $outputRoot = "geolifeclef-risk-aware-sdm/artifacts/$ArtifactDirectory"
    $submissionEntry = $outputs.files | Where-Object {
        $_.fileName -eq "$outputRoot/GLC25_PA_submission.csv"
    } | Select-Object -First 1
    $manifestEntry = $outputs.files | Where-Object {
        $_.fileName -eq "$outputRoot/$ManifestName"
    } | Select-Object -First 1
    if (-not $submissionEntry -or -not $manifestEntry) {
        throw "The completed kernel does not contain both validated submission outputs."
    }
    & curl.exe -L --fail --silent --show-error $submissionEntry.url -o $submissionFile
    if ($LASTEXITCODE -ne 0) { throw "Could not retrieve the generated submission CSV." }
    & curl.exe -L --fail --silent --show-error $manifestEntry.url -o $manifestFile.FullName
    if ($LASTEXITCODE -ne 0) { throw "Could not retrieve the generated submission manifest." }

    $manifest = Get-Content -LiteralPath $manifestFile.FullName -Raw | ConvertFrom-Json
    $digest = (Get-FileHash -LiteralPath $submissionFile -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($digest -ne $manifest.submission.submission_sha256) {
        throw "Submission SHA-256 does not match the kernel manifest."
    }
    if ($manifest.test_labels_used -ne $false) { throw "Manifest does not prove test-label isolation." }
    if ([int]$manifest.submission.rows -ne 14784) { throw "Manifest has an unexpected row count." }
    if ($AdaptivePolicy) {
        if ([double]$manifest.geographic_holdout.tuning_optimized_f1 -lt [double]$manifest.geographic_holdout.tuning_neural_only_adaptive_f1) {
            throw "The tuned adaptive policy underperformed its neural-only control."
        }
    }
    elseif ([int]$manifest.submission.prediction_policy.k -ne 18) {
        throw "Manifest does not use the registered top-18 policy."
    }
    $rows = Import-Csv -LiteralPath $submissionFile
    if ($rows.Count -ne 14784) { throw "Submission CSV has an unexpected row count." }
    if ((($rows.surveyId | Sort-Object -Unique).Count) -ne 14784) {
        throw "Submission CSV contains duplicate or missing survey IDs."
    }
    $predictionCounts = @($rows | ForEach-Object { @($_.predictions -split '\s+' | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }).Count })
    if (@($predictionCounts | Where-Object { $_ -lt 1 }).Count -ne 0) {
        throw "Submission CSV contains an empty prediction row."
    }
    if ($AdaptivePolicy) {
        if (@($predictionCounts | Where-Object { $_ -gt 50 }).Count -ne 0) {
            throw "Adaptive submission exceeds its registered maximum cardinality."
        }
    }
    elseif (@($predictionCounts | Where-Object { $_ -ne 18 }).Count -ne 0) {
        throw "Submission CSV contains a row that is not top-18."
    }

    & curl.exe -L --fail --silent --show-error `
        -H "Authorization: Basic $auth" `
        "https://www.kaggle.com/api/v1/competitions/data/download/$Competition/GLC25_SAMPLE_SUBMISSION.csv" `
        -o $templateFile.FullName
    if ($LASTEXITCODE -ne 0) { throw "Could not retrieve the official submission template." }
    $templateRows = Import-Csv -LiteralPath $templateFile.FullName
    $submittedIds = @($rows.surveyId | Sort-Object)
    $templateIds = @($templateRows.surveyId | Sort-Object)
    if ((Compare-Object -ReferenceObject $templateIds -DifferenceObject $submittedIds).Count -ne 0) {
        throw "Submission survey IDs do not exactly match the official template."
    }

    $clientRoot = Join-Path $PSScriptRoot "..\.venv\kaggle-api"
    $client = Join-Path $clientRoot "bin\kaggle.exe"
    if (-not (Test-Path -LiteralPath $client)) {
        throw "The official Kaggle API client is not installed in the ignored runtime directory."
    }
    $previousPythonPath = [Environment]::GetEnvironmentVariable("PYTHONPATH")
    $previousUsername = [Environment]::GetEnvironmentVariable("KAGGLE_USERNAME")
    $previousKey = [Environment]::GetEnvironmentVariable("KAGGLE_KEY")
    try {
        [Environment]::SetEnvironmentVariable("PYTHONPATH", (Resolve-Path $clientRoot).Path)
        [Environment]::SetEnvironmentVariable("KAGGLE_USERNAME", $credential.username)
        [Environment]::SetEnvironmentVariable("KAGGLE_KEY", $credential.key)
        $clientOutput = & $client competitions submit $Competition -f $submissionFile -m $Message -q 2>&1
        if ($LASTEXITCODE -ne 0) {
            $detail = ($clientOutput | Out-String).Trim()
            $detail = [regex]::Replace($detail, '(?i)(token|key|password|authorization)\s*[:=]\s*[^,}\s]+', '$1=[redacted]')
            throw "Kaggle competition submission failed: $detail"
        }
    }
    finally {
        [Environment]::SetEnvironmentVariable("PYTHONPATH", $previousPythonPath)
        [Environment]::SetEnvironmentVariable("KAGGLE_USERNAME", $previousUsername)
        [Environment]::SetEnvironmentVariable("KAGGLE_KEY", $previousKey)
    }
    [pscustomobject]@{
        Status = "submitted"
        Competition = $Competition
        Kernel = $KernelSlug
        Rows = 14784
        PredictionCountMinimum = [int](($predictionCounts | Measure-Object -Minimum).Minimum)
        PredictionCountMaximum = [int](($predictionCounts | Measure-Object -Maximum).Maximum)
        PredictionCountMean = [double](($predictionCounts | Measure-Object -Average).Average)
        Sha256Verified = $true
        TemplateIdsVerified = $true
        TestLabelsUsed = $false
        AdaptivePolicy = [bool]$AdaptivePolicy
    } | ConvertTo-Json -Compress
}
finally {
    Remove-Item -LiteralPath $metadataFile.FullName -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $submissionFile -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $manifestFile.FullName -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $templateFile.FullName -Force -ErrorAction SilentlyContinue
}
