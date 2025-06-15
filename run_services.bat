@echo off
echo Starting Salary Predictor Services...

call venv\Scripts\activate.bat

echo Starting FastAPI server...
start "FastAPI Server" cmd /k "python api_app.py"

timeout /t 5 /nobreak > nul

echo Starting Streamlit app...
start "Streamlit App" cmd /k "streamlit run streamlit_app.py"

echo Both services are starting...
echo FastAPI will be available at: http://localhost:8000
echo Streamlit will be available at: http://localhost:8501
echo.
echo Press any key to stop all services...
pause > nul

echo Stopping services...
taskkill /f /im python.exe > nul 2>&1
echo Services stopped.
pause
