@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem === CONFIG ===
set "DOCKER_PATH=C:\Program Files\Docker\Docker\Docker Desktop.exe"
set "COMPOSE_FILE=../docker-compose.prod.yaml"
rem === /CONFIG ===

rem Script's directory (always reliable, regardless of caller CWD)
set "SCRIPT_DIR=%~dp0"

echo Script dir: "%SCRIPT_DIR%"
echo Caller CWD: "%CD%"

rem Ensure the compose file exists (absolute path)
if not exist "%SCRIPT_DIR%%COMPOSE_FILE%" (
    echo ERROR: "%SCRIPT_DIR%%COMPOSE_FILE%" not found.
    exit /b 2
)

rem (Optional but nice) make Compose treat the project as if run from the script dir
set "COMPOSE_PROJECT_DIR=%SCRIPT_DIR%"

rem Switch to the script dir (handles drive changes)
cd /d "%SCRIPT_DIR%" || ( echo ERROR: cannot cd to "%SCRIPT_DIR%" & exit /b 3 )

rem --- 1) Ensure Docker Desktop is running ---
tasklist /FI "IMAGENAME eq Docker Desktop.exe" | find /I "Docker Desktop.exe" >nul
if errorlevel 1 (
    echo Starting Docker Desktop...
    start "" "%DOCKER_PATH%"
) else (
    echo Docker Desktop is already running.
)

rem --- 2) Wait for Docker engine to be ready ---
echo Waiting for Docker Engine to start...
:waitloop
docker info >nul 2>&1 || ( timeout /t 2 /nobreak >nul & goto waitloop )
echo Docker Engine is ready!

rem --- 3) Prefer docker compose (v2); fallback to docker-compose (v1) ---
docker compose version >nul 2>&1
if %ERRORLEVEL%==0 ( set "DC_CMD=docker compose" ) else ( set "DC_CMD=docker-compose" )

echo Using: %DC_CMD%
echo Compose file: "%SCRIPT_DIR%%COMPOSE_FILE%"

rem --- 4) Bring the stack up (absolute -f path so caller CWD never matters) ---
%DC_CMD% -f "%SCRIPT_DIR%%COMPOSE_FILE%" up
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" (
    echo Compose failed with exit code %RC%.
) else (
    echo Project started successfully.
)

exit %RC%   rem <- closes the cmd window created to run this BAT
