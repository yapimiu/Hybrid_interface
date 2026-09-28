@echo off
setlocal
cd /d "%~dp0"
py -3 -m venv .venv-windows
if errorlevel 1 goto failed
call .venv-windows\Scripts\activate.bat
python -m pip install -r requirements.txt -r requirements-dev.txt
if errorlevel 1 goto failed
python scripts\build_app.py
if errorlevel 1 goto failed
echo.
echo Built: dist\HybridInterface.exe
pause
exit /b 0
:failed
echo.
echo Build failed. See the error above.
pause
exit /b 1
