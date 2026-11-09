@echo off
setlocal
cd /d "%~dp0\.."
if not exist ".venv\Scripts\pythonw.exe" (
    echo Missing project virtual environment. Follow the installation instructions in README.md.
    pause
    exit /b 1
)
rem pythonw runs without a console window; the batch window closes immediately.
rem Use run_debug.bat to see tracebacks printed before the window appears.
start "" ".venv\Scripts\pythonw.exe" -m pySAview %*
