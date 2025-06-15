@echo off
echo.
echo ================================================
echo     Salary Predictor FastAPI Project
echo ================================================
echo.

cd /d "c:\Users\Zeshan\Desktop\Salary pridictor fast api project"

echo [1/4] Activating virtual environment...
call venv\Scripts\activate.bat

echo [2/4] Testing FastAPI server...
python test_services.py

echo.
echo [3/4] Services Status:
echo   ✅ FastAPI Server: http://localhost:8001
echo   ✅ API Documentation: http://localhost:8001/docs
echo   ❗ Streamlit: Manual start required
echo.

echo [4/4] To start Streamlit manually:
echo   1. Open a NEW command prompt
echo   2. Run: cd "c:\Users\Zeshan\Desktop\Salary pridictor fast api project"
echo   3. Run: venv\Scripts\activate.bat
echo   4. Run: streamlit run streamlit_app.py
echo.

echo ================================================
echo  🎉 FastAPI is LIVE and ready to use!
echo ================================================
echo.
echo Available endpoints:
echo   • Health Check: GET http://localhost:8001/
echo   • Predict Salary: POST http://localhost:8001/predict_salary
echo   • Interactive Docs: http://localhost:8001/docs
echo.

echo Press any key to open the API documentation in your browser...
pause > nul
start http://localhost:8001/docs

echo.
echo Keep this window open to maintain the FastAPI server.
echo Press Ctrl+C to stop the server.
echo.
pause
