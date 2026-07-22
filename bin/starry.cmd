@echo off
rem Starry Code launcher (Windows). Forwards all args to starry.py.
setlocal
set "SCRIPT_DIR=%~dp0"
python "%SCRIPT_DIR%starry.py" %*
exit /b %ERRORLEVEL%
