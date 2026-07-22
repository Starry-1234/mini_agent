@echo off
rem Starry Code launcher (Windows / cmd.exe / PowerShell-compatible).
rem Works from any directory: resolves the project root from this
rem script's own location, then runs `docker compose run --rm agent`
rem with all forwarded arguments.
rem
rem Install: add this folder to PATH, e.g.
rem     setx PATH "%PATH%;F:\dev\AI_Tools\workspace\mini_agent\bin"
rem Or drop a shortcut / mklink of starry.cmd into a folder already on PATH.

setlocal
set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%..\"
pushd "%PROJECT_ROOT%" >nul || (echo [starry] could not cd to "%PROJECT_ROOT%" 1>&2 & exit /b 1)
docker compose run --rm agent %*
set "EXITCODE=%ERRORLEVEL%"
popd >nul
exit /b %EXITCODE%
