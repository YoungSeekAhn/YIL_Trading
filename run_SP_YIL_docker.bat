@echo off
setlocal

cd /d C:\Users\ganys\python_work\YIL_trading

REM (중요) 한 번 실행하고 컨테이너 삭제
docker compose run --rm yil_trading >> logs\docker_run.log 2>&1