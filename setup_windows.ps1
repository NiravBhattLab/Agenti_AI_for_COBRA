<#
Creates the agentic_cobra conda environment on Windows and installs the
prodigal / diamond binaries that bioconda does not ship for win-64.

bioconda only builds `prodigal` (used by mackinac/carveme for gene calling)
and `diamond` (used by carveme for homology search) for linux-64/osx-64.
Both projects publish official precompiled Windows binaries directly on
GitHub, so this script downloads those and drops them in the environment's
Library\bin, which conda puts on PATH whenever the environment is active.

Usage:
  powershell -ExecutionPolicy Bypass -File setup_windows.ps1
#>

$ErrorActionPreference = "Stop"

$EnvName = "agentic_cobra"
$EnvYml = Join-Path $PSScriptRoot "environment-windows.yml"

$ProdigalUrl = "https://github.com/hyattpd/Prodigal/releases/download/v2.6.3/prodigal.windows.exe"
$DiamondZipUrl = "https://github.com/bbuchfink/diamond/releases/download/v2.2.5/diamond-windows.zip"

function Find-CondaExe {
    if ($env:CONDA_EXE -and (Test-Path $env:CONDA_EXE)) {
        return $env:CONDA_EXE
    }
    $onPath = Get-Command conda.exe -ErrorAction SilentlyContinue
    if ($onPath) {
        return $onPath.Source
    }
    $candidates = @(
        "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
        "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
        "$env:LOCALAPPDATA\anaconda3\Scripts\conda.exe",
        "$env:LOCALAPPDATA\miniconda3\Scripts\conda.exe",
        "C:\ProgramData\Anaconda3\Scripts\conda.exe",
        "C:\ProgramData\miniconda3\Scripts\conda.exe"
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) { return $c }
    }
    throw "Could not find conda.exe. Activate your conda install (or set `$env:CONDA_EXE`) and re-run."
}

$CondaExe = Find-CondaExe
Write-Host "Using conda: $CondaExe"

$existingEnvs = & $CondaExe env list | Out-String
if ($existingEnvs -match "(?m)^\s*$EnvName\s") {
    Write-Host "Environment '$EnvName' already exists, skipping creation. Delete it first (conda env remove -n $EnvName) to recreate from scratch."
} else {
    Write-Host "Creating conda environment from $EnvYml ..."
    & $CondaExe env create -f $EnvYml
    if ($LASTEXITCODE -ne 0) { throw "conda env create failed with exit code $LASTEXITCODE" }
}

$CondaBase = (& $CondaExe info --base).Trim()
$LibraryBin = Join-Path $CondaBase "envs\$EnvName\Library\bin"
New-Item -ItemType Directory -Force -Path $LibraryBin | Out-Null

$TempDir = Join-Path $env:TEMP "agentic_cobra_setup"
New-Item -ItemType Directory -Force -Path $TempDir | Out-Null

$ProdigalDest = Join-Path $LibraryBin "prodigal.exe"
if (Test-Path $ProdigalDest) {
    Write-Host "prodigal.exe already present, skipping download."
} else {
    Write-Host "Downloading prodigal.exe ..."
    Invoke-WebRequest -Uri $ProdigalUrl -OutFile $ProdigalDest
}

$DiamondDest = Join-Path $LibraryBin "diamond.exe"
if (Test-Path $DiamondDest) {
    Write-Host "diamond.exe already present, skipping download."
} else {
    Write-Host "Downloading diamond.exe ..."
    $DiamondZip = Join-Path $TempDir "diamond-windows.zip"
    Invoke-WebRequest -Uri $DiamondZipUrl -OutFile $DiamondZip
    Expand-Archive -Path $DiamondZip -DestinationPath $TempDir -Force
    $DiamondExe = Get-ChildItem -Path $TempDir -Filter "diamond.exe" -Recurse | Select-Object -First 1
    if (-not $DiamondExe) { throw "diamond.exe not found inside diamond-windows.zip" }
    Copy-Item $DiamondExe.FullName $DiamondDest -Force
}

Remove-Item -Recurse -Force $TempDir -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "Verifying binaries:"
& $ProdigalDest -v
& $DiamondDest version

Write-Host ""
Write-Host "Done. Activate with: conda activate $EnvName"
