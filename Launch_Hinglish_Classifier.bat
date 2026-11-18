@echo off
setlocal
cd /d "%~dp0"

REM Use the project's virtual environment when available.
if exist "%~dp0.venv\Scripts\python.exe" (
    set "PYTHON=%~dp0.venv\Scripts\python.exe"
) else if exist "%~dp0venv\Scripts\python.exe" (
    set "PYTHON=%~dp0venv\Scripts\python.exe"
) else (
    set "PYTHON=python"
)

echo Starting Hinglish NLP Classifier...
"%PYTHON%" "%~dp0hinglish_desktop_app.py"
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
    echo.
    echo The application closed with an error. Check the message above.
) else (
    echo.
    echo Hinglish NLP Classifier closed.
)
pause
exit /b %EXIT_CODE%
