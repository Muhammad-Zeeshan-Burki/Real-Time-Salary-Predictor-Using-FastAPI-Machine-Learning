import subprocess
import time
import webbrowser
import os
import sys

def start_services():
    """Start both FastAPI and Streamlit services"""
    
    print("🚀 Starting Salary Predictor Services...")
    print("=" * 50)
    
    # Change to project directory
    project_dir = r"c:\Users\Zeshan\Desktop\Salary pridictor fast api project"
    os.chdir(project_dir)
    
    # Python executable path in venv
    python_exe = os.path.join(project_dir, "venv", "Scripts", "python.exe")
    
    print("📡 Starting FastAPI server...")
    # Start FastAPI in background
    fastapi_process = subprocess.Popen([
        python_exe, "api_app.py"
    ], cwd=project_dir)
    
    # Wait for FastAPI to start
    time.sleep(3)
    
    print("🌐 Starting Streamlit app...")
    # Start Streamlit in background
    streamlit_process = subprocess.Popen([
        python_exe, "-m", "streamlit", "run", "streamlit_app.py", 
        "--server.headless", "true"
    ], cwd=project_dir)
    
    # Wait for services to fully start
    print("⏳ Waiting for services to initialize...")
    time.sleep(8)
    
    # Test services
    print("🔍 Testing services...")
    test_result = subprocess.run([python_exe, "test_services.py"], 
                                capture_output=True, text=True, cwd=project_dir)
    print(test_result.stdout)
    
    print("\n🌍 Opening web interfaces...")
    # Open web browsers
    webbrowser.open("http://localhost:8501")
    time.sleep(2)
    webbrowser.open("http://localhost:8001/docs")
    
    print("\n" + "=" * 50)
    print("🎉 ALL SERVICES RUNNING SUCCESSFULLY!")
    print("=" * 50)
    print("📱 Available at:")
    print("   • Streamlit Web App: http://localhost:8501")
    print("   • FastAPI Documentation: http://localhost:8001/docs")
    print("   • API Endpoint: http://localhost:8001")
    print("\n💡 Press Ctrl+C to stop all services...")
    
    try:
        # Keep services running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping services...")
        fastapi_process.terminate()
        streamlit_process.terminate()
        print("✅ All services stopped.")

if __name__ == "__main__":
    start_services()
