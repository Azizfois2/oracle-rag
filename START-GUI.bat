@echo off
rem ---------------------------------------------------------------------------
rem Oracle RAG Console - one-click launcher (Streamlit on port 8501).
rem
rem Behaviour:
rem   1. verifies the "rag" virtual environment and the app dependencies
rem   2. reuses a console that is already answering on port 8501
rem   3. otherwise starts Streamlit, waits for it, then opens the browser
rem   4. writes server output to logs\gui-YYYYMMDD.log
rem
rem Keep this window open while you use the console - closing it stops the server.
rem Set RAG_NO_BROWSER=1 to print the URL instead of opening a browser.
rem ---------------------------------------------------------------------------
setlocal EnableExtensions
title Oracle RAG Console

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
set "PY=%USERPROFILE%\rag-venv\Scripts\python.exe"
set "PORT=8501"
set "URL=http://localhost:%PORT%"
set "LOGDIR=%ROOT%\logs"

if not exist "%PY%" goto :no_venv
if not exist "%LOGDIR%" mkdir "%LOGDIR%"
for /f "usebackq delims=" %%d in (`powershell -NoProfile -Command "Get-Date -Format yyyyMMdd"`) do set "TODAY=%%d"
if not defined TODAY set "TODAY=session"
set "LOG=%LOGDIR%\gui-%TODAY%.log"

rem --- dependencies -----------------------------------------------------------
"%PY%" -c "import streamlit, pypdf" >nul 2>&1
if not errorlevel 1 goto :check_running

echo Installing the console dependencies, this takes a minute the first time...
echo.
"%PY%" -m pip install -r "%ROOT%\requirements.txt"
if errorlevel 1 goto :no_deps
"%PY%" -c "import streamlit, pypdf" >nul 2>&1
if errorlevel 1 goto :no_deps

:check_running
rem --- reuse a server that is already up --------------------------------------
powershell -NoProfile -Command "try{$r=Invoke-WebRequest '%URL%/_stcore/health' -UseBasicParsing -TimeoutSec 3; if($r.StatusCode -eq 200){exit 0}else{exit 1}}catch{exit 1}" >nul 2>&1
if errorlevel 1 goto :launch

echo The console is already running.
echo   %URL%
if defined RAG_NO_BROWSER goto :done
start "" "%URL%"
goto :done

:launch
echo.
echo   Oracle RAG Console
echo   address : %URL%
echo   log     : %LOG%
echo.
echo Server starting, the browser opens as soon as it answers.
echo Keep this window open while you use the console - closing it stops the server.
echo.

if defined RAG_NO_BROWSER goto :serve
start "" /b powershell -NoProfile -WindowStyle Hidden -Command "for($i=0;$i -lt 120;$i++){$u='%URL%';try{if((Invoke-WebRequest ($u + '/_stcore/health') -UseBasicParsing -TimeoutSec 2).StatusCode -eq 200){Start-Process $u;break}}catch{};Start-Sleep -Seconds 1}"

:serve
cd /d "%ROOT%"
"%PY%" -m streamlit run "%ROOT%\app.py" --server.port %PORT% --server.headless true >>"%LOG%" 2>&1

set "CODE=%ERRORLEVEL%"
echo.
echo The console has stopped.
if not "%CODE%"=="0" (
  echo Exit code %CODE%. The last lines of the log:
  echo.
  powershell -NoProfile -Command "Get-Content '%LOG%' -Tail 25"
)
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

:no_deps
echo ERROR: the app dependencies could not be installed.
echo Install them by hand with
echo   "%PY%" -m pip install -r "%ROOT%\requirements.txt"
echo and double-click this launcher again.
pause

:done
endlocal
