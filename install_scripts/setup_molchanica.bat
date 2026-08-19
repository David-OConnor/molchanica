@echo off
rem Double-click launcher for setup_molchanica.ps1. Runs it without requiring a change to the
rem PowerShell execution policy, which otherwise blocks scripts extracted from a downloaded zip.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_molchanica.ps1" %*
pause
