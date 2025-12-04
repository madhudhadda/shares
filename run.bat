@echo off
:: This batch file automates the startup of the AI Investment Strategist app.

:: Set the title of the command prompt window
title AI Investment Strategist

echo ==========================================================
echo  Starting the AI Investment Strategist Application...
echo ==========================================================
echo.

:: 1. Navigate to the project directory.
:: The /d switch is important as it changes the drive as well as the directory.
cd /d "F:\both_investor"
echo [OK] Navigated to project directory: %cd%

echo.

:: 2. Activate the Python virtual environment.
:: We use 'call' to ensure the script continues after activation.
echo [INFO] Activating the virtual environment...
call venv\Scripts\activate

echo.

:: 3. Run the Streamlit application.
echo [INFO] Starting the Streamlit server...
echo [INFO] Your browser should open shortly.
echo.
streamlit run app2.py

echo.
echo ==========================================================
echo  The Streamlit server has been started.
echo  To stop the server, simply close this window.
echo ==========================================================
echo.

:: Keep the command window open after the script finishes or if an error occurs.
pause