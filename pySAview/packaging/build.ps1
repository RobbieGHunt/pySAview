# Build script for SuperADAM NeXus Explorer standalone windowed executable
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Split-Path -Parent (Split-Path -Parent $ScriptDir)

Write-Host "Building SuperADAM Explorer standalone executable with PyInstaller..." -ForegroundColor Cyan

# Clean previous build artifacts
$BuildDir = Join-Path $RootDir "build\pySAview"
$DistRoot = Join-Path $RootDir "dist"
$BundleDir = Join-Path $DistRoot "pySAview"
if (Test-Path $BuildDir) { Remove-Item -Recurse -Force $BuildDir }
if (Test-Path $BundleDir) { Remove-Item -Recurse -Force $BundleDir }

$PyInstaller = Join-Path $RootDir ".venv\Scripts\pyinstaller.exe"
$SpecFile = Join-Path $ScriptDir "pySAview.spec"

& $PyInstaller --clean --noconfirm --distpath $DistRoot --workpath $BuildDir $SpecFile

$ExePath = Join-Path $BundleDir "pySAview.exe"
if (-not (Test-Path $ExePath)) {
    Write-Error "Build failed: $ExePath not found!"
    exit 1
}

$totalBytes = (Get-ChildItem -Recurse -File $BundleDir | Measure-Object -Property Length -Sum).Sum
$sizeMB = [math]::Round($totalBytes / 1MB, 2)
Write-Host "Build complete! Bundle: $BundleDir (Total size: $sizeMB MB)" -ForegroundColor Green
