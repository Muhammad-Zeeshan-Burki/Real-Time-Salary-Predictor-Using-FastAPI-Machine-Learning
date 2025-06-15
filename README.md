# Real‑Time Salary Predictor API 💼

A production-ready FastAPI application for predicting software developer salaries in **real-time**, powered by a trained machine learning model.

---

## 🚀 Project Overview

- **Live API** to instantly estimate a developer’s annual salary based on professional attributes.
- Built with **FastAPI** for high-performance REST endpoints and **automatic documentation**.
- Utilizes a **serialized ML model** (e.g., `scikit-learn`, `joblib`) for inference.
- Ideal for integration into dashboards, career tools, HR platforms, or as a standalone service.

---

## 📁 Project Structure

/
├── app/
│ ├── main.py # FastAPI application instance
│ ├── models.py # Pydantic schemas for request/response models
│ ├── predictor.py # Loads model and runs inference
│ ├── utils.py # Preprocessing, encoders, feature transforms
│ └── ...
├── data/
│ ├── salary_data.csv # Raw training data
│ ├── train.py # Script to clean, train, and serialize model
│ └── ...
├── artifacts/
│ ├── model.joblib # Trained ML model
│ ├── encoder.joblib # Label or OneHot encoder
├── .env # Environment variables (e.g. port, MODEL_PATH)
├── requirements.txt
└── README.md

---

## 🎯 Key Features

- **`/health`** – GET endpoint for simple health and readiness checks.
- **`/predict`** – POST endpoint to obtain salary estimations from JSON input.
- **API docs** powered by Swagger UI: visit `/docs` or `/redoc` after running.
- **Modular ML pipeline**: preprocess → inference → return result.
- **Scalable & Docker-ready**: supports containerization for easy deployment.

---

## 🛠️ Setup & Usage

### Requirements

- Python 3.9+  
- Virtualenv or Conda  

### Installation

```bash
git clone https://github.com/Muhammad-Zeeshan-Burki/Real-Time-Salary-Predictor-Using-FastAPI-Machine-Learning.git
cd Real-Time-Salary-Predictor-Using-FastAPI-Machine-Learning
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
'''
