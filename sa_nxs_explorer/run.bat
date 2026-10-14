@echo off
cd /d "%~dp0\.."
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m sa_nxs_explorer %*
) else (
    python -m sa_nxs_explorer %*
)
pause
