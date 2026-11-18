@echo off
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0Push_To_GitHub.ps1"
if errorlevel 1 (
  echo.
  echo The GitHub upload did not complete. Read the message above.
) 
echo.
pause
