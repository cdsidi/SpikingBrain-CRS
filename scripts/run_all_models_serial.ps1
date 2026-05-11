param(
    [int]$Epochs = 20,
    [int]$BatchSize = 4,
    [int]$Folds = 5,
    [int]$Repeats = 5,
    [string]$BaseOutput = "results",
    [switch]$AllDatasets = $true
)

$ErrorActionPreference = "Stop"

function Run-Model {
    param(
        [string]$ModelName
    )

    $outDir = Join-Path $BaseOutput ("full_protocol_" + $ModelName)

    Write-Host "===============================================================" -ForegroundColor Cyan
    Write-Host "Starting model: $ModelName" -ForegroundColor Green
    Write-Host "Output dir    : $outDir"
    Write-Host "Epochs/Folds/Repeats/BS: $Epochs/$Folds/$Repeats/$BatchSize"
    Write-Host "===============================================================" -ForegroundColor Cyan

    if ($AllDatasets) {
        $cmd = "python -m training.train_crs --all-datasets --model $ModelName --epochs $Epochs --batch-size $BatchSize --folds $Folds --repeats $Repeats --output-dir $outDir"
    }
    else {
        throw "This script currently expects -AllDatasets mode."
    }

    Write-Host "Running: $cmd" -ForegroundColor Yellow
    Invoke-Expression $cmd

    if ($LASTEXITCODE -ne 0) {
        throw "Model $ModelName failed with exit code $LASTEXITCODE"
    }

    Write-Host "Completed model: $ModelName" -ForegroundColor Green
}

$models = @("t_base", "t_crs", "s_base", "s_crs")

$start = Get-Date
Write-Host "Serial run started at: $start" -ForegroundColor Magenta

foreach ($m in $models) {
    Run-Model -ModelName $m
}

$end = Get-Date
$elapsed = $end - $start

Write-Host "===============================================================" -ForegroundColor Cyan
Write-Host "All models completed successfully." -ForegroundColor Green
Write-Host "Start: $start"
Write-Host "End  : $end"
Write-Host "Elapsed: $($elapsed.ToString())"
Write-Host "===============================================================" -ForegroundColor Cyan

