@echo off
setlocal
cd /d "%~dp0\.."
if not exist ".venv\Scripts\python.exe" (
    echo Missing project virtual environment. Follow the installation instructions in README.md.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m sa_nxs_explorer %*
pause
