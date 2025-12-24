@echo off
setlocal

cd /d C:\Users\ganys\python_work\YIL_trading

if not exist logs mkdir logs

"C:\Program Files\Python313\python.exe" SP_YIL.py >> logs\SP_YIL.log 2>&1
