@echo off
rem ---------------------------------------------------------------------------
rem Oracle RAG Console - full-screen terminal interface.
rem
rem Checks the rag virtual environment and the Textual dependency, then starts
rem tui.py in this window. Keys: ctrl+q quit, ctrl+t theme, ctrl+s save artefact.
rem Pass arguments through, for example:
rem   START-TUI.bat --local
rem   START-TUI.bat --theme light --no-history
rem ---------------------------------------------------------------------------
setlocal EnableExtensions
title Oracle RAG Console - TUI

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
set "PY=%USERPROFILE%\rag-venv\Scripts\python.exe"

if not exist "%PY%" goto :no_venv
if not exist "%ROOT%\tui.py" goto :no_app

"%PY%" -c "import textual, rich" >nul 2>&1
if not errorlevel 1 goto :launch

echo Installing the terminal interface dependencies...
echo.
"%PY%" -m pip install -r "%ROOT%\requirements.txt"
if errorlevel 1 goto :no_deps

:launch
cd /d "%ROOT%"
"%PY%" "%ROOT%\tui.py" %*
set "CODE=%ERRORLEVEL%"
if "%CODE%"=="0" goto :done

echo.
echo The TUI stopped with exit code %CODE%.
echo Check the Oracle container and .env, then try again.
echo.
pause
goto :done

:no_venv
echo ERROR: the Python environment was not found at
echo   %PY%
echo.
echo Restore it with START.bat from the rag-oneclick folder, or create it with
echo   py -3.11 -m venv "%USERPROFILE%\rag-venv"
echo Then double-click this launcher again.
pause
goto :done

:no_app
echo ERROR: tui.py was not found next to this launcher.
pause
goto :done

:no_deps
echo ERROR: the terminal dependencies could not be installed.
echo Install them by hand with
echo   "%PY%" -m pip install -r "%ROOT%\requirements.txt"
pause
goto :done

:done
endlocal
