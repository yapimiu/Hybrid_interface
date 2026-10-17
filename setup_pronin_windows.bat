@echo off
setlocal
cd /d "%~dp0"
py -3.10 -m venv .venv-pronin
if errorlevel 1 goto failed
.venv-pronin\Scripts\python.exe -m pip install -r requirements-pronin.txt
if errorlevel 1 goto failed
.venv-pronin\Scripts\python.exe scripts\install_lsl.py
if errorlevel 1 goto failed
echo Ready: select .venv-pronin\Scripts\python.exe in the Launch tab.
pause
exit /b 0
:failed
echo Setup failed. Install Python 3.10 with the Python Launcher, then retry.
pause
exit /b 1
