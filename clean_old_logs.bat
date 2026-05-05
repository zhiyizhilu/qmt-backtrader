@echo off
echo Cleaning old log files in logs directory...

set "LOG_DIR=%~dp0logs"

if not exist "%LOG_DIR%" (
    echo Log directory does not exist: %LOG_DIR%
    exit /b 1
)

forfiles /P "%LOG_DIR%" /S /M *.* /D -1 /C "cmd /c del /F /Q @path" 2>nul

if %errorlevel% equ 0 (
    echo Cleanup completed!
) else (
    echo No files to clean or error occurred.
)

pause
