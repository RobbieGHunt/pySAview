# Build script for SuperADAM NeXus Explorer standalone windowed executable
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Split-Path -Parent (Split-Path -Parent $ScriptDir)

Write-Host "Building SuperADAM Explorer standalone executable with PyInstaller..." -ForegroundColor Cyan

# Clean previous build artifacts
$BuildDir = Join-Path $RootDir "build\sa_nxs"
$DistDir = Join-Path $RootDir "dist\sa_nxs"
if (Test-Path $BuildDir) { Remove-Item -Recurse -Force $BuildDir }
if (Test-Path $DistDir) { Remove-Item -Recurse -Force $DistDir }

$PyInstaller = Join-Path $RootDir ".venv\Scripts\pyinstaller.exe"
$SpecFile = Join-Path $ScriptDir "sa_nxs_explorer.spec"

& $PyInstaller --clean --distpath $DistDir --workpath $BuildDir $SpecFile

$ExePath = Join-Path $DistDir "SuperADAM_Explorer.exe"
if (-not (Test-Path $ExePath)) {
    Write-Error "Build failed: $ExePath not found!"
    exit 1
}

$exe = Get-Item $ExePath
$sizeMB = [math]::Round($exe.Length / 1MB, 2)
Write-Host "Build complete! Executable: $($exe.FullName) (Size: $sizeMB MB)" -ForegroundColor Green
