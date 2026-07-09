@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\pythonw.exe" (
    start "" ".venv\Scripts\pythonw.exe" "Paint\paint_ui.py"
) else (
    python "Paint\paint_ui.py"
)
endlocal
