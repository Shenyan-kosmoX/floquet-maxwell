param(
    [ValidateSet('lpor','science')][string]$Case = 'lpor',
    [int]$Resolution = 24,
    [double]$Kx = 0,
    [double]$Epsilon = 16,
    [double]$Padding = 1,
    [double]$WindowTime = 100,
    [string]$OutputName = 'local_probe'
)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$driveLetter = ([System.IO.Path]::GetPathRoot($taskRoot)).Substring(0,1).ToLowerInvariant()
$linuxRoot = '/mnt/' + $driveLetter + ($taskRoot.Substring(2) -replace '\\','/')
if ($OutputName -notmatch '^[a-zA-Z0-9_.-]+$') { throw 'OutputName must be a simple folder name.' }
$outputPath = 'toolbox_experiments/results/local/' + $OutputName
$invariant = [System.Globalization.CultureInfo]::InvariantCulture
& wsl.exe -d Ubuntu-24.04 --cd $linuxRoot -- env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 `
    /home/codex/.local/ptc-meep-env/bin/python toolbox_experiments/scripts/meep_mode_probe.py `
    --case $Case --resolution $Resolution --kx $Kx.ToString($invariant) `
    --epsilon $Epsilon.ToString($invariant) --padding $Padding.ToString($invariant) `
    --window-time $WindowTime.ToString($invariant) --output $outputPath
if ($LASTEXITCODE -ne 0) { throw "Meep probe failed with exit code $LASTEXITCODE" }
Write-Host ('Results: ' + (Join-Path $taskRoot ($outputPath -replace '/','\')))
