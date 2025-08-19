@echo off
cd /d "%~dp0" || exit /b 3
docker compose -f "../docker-compose.prod.yaml" down
exit %RC%   rem <- closes the cmd window created to run this BAT