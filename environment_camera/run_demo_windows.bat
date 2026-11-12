@echo off
setlocal
cd /d "%~dp0"
set "HOST=127.0.0.1"
set "DEMO_MODE=1"
set "ADMIN_PIN=DEMO-ONLY-2468"
echo ===============================================
echo  HAZARDSCOUT / OFFLINE DEMO MODE ONLY
echo  Dashboard: http://127.0.0.1:8000
echo  Demo review PIN: DEMO-ONLY-2468
echo  No USB camera or real YOLO inference is used.
echo ===============================================
if exist "..\.venv\Scripts\python.exe" (
    "..\.venv\Scripts\python.exe" server.py
) else if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" server.py
) else (
    python server.py
)
pause
