# Creates a clean Windows Desktop / Folder Shortcut for SuperADAM NeXus Explorer
# Points directly to pythonw.exe so no black command prompt window ever appears.

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Split-Path -Parent $ScriptDir

$Pythonw = Join-Path $RootDir ".venv\Scripts\pythonw.exe"
if (-not (Test-Path $Pythonw)) {
    $Pythonw = "pythonw.exe"
}

$ShortcutPath = Join-Path $ScriptDir "SuperADAM_Explorer.lnk"

$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = $Pythonw
$Shortcut.Arguments = "-m sa_nxs_explorer"
$Shortcut.WorkingDirectory = $RootDir
$Shortcut.Description = "SuperADAM NeXus Scan File Explorer"
$Shortcut.Save()

Write-Host "Created native Windows shortcut: $ShortcutPath" -ForegroundColor Green
Write-Host "Target: $Pythonw -m sa_nxs_explorer" -ForegroundColor Cyan
