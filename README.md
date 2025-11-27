# 📉 Customer Churn Prediction – End-to-End ML Pipeline

![Uploading churn  image.png…]()


An end-to-end **Machine Learning system** that predicts telecom customer churn and exposes the model through a **FastAPI** service with a **Streamlit** dashboard for business users.

This project demonstrates the **full ML lifecycle**:

- Data ingestion & preprocessing  
- Model training and selection  
- Experiment tracking with MLflow  
- Model serving via REST API  
- Interactive UI for predictions & insights  
- (Optional) Dockerized architecture

---

## 🎯 Problem Statement

Telecom companies lose significant revenue when customers discontinue their services (“churn”).  
The goal of this project is to **predict the probability of churn for each customer**, so the business can:

- Identify **high-risk** customers  
- Design **retention campaigns**  
- Reduce **revenue loss** and improve **lifetime value**

---

## 🧠 Key Features

- ✅ End-to-end ML pipeline from raw data → deployed model  
- ✅ Preprocessing & feature engineering using **Pandas + Scikit-learn**  
- ✅ Multiple models: **Logistic Regression, Random Forest, XGBoost**  
- ✅ Model selection based on **AUC, F1, Precision, Recall**  
- ✅ Experiment tracking with **MLflow**  
- ✅ Model persisted as a **single pickle pipeline** (`best_model.pkl`)  
- ✅ **FastAPI** for real-time prediction via `/predict` endpoint  
- ✅ **Streamlit dashboard** for:
  - Single-customer prediction
  - Basic churn analytics (distribution, churn vs contract, etc.)
- ✅ Designed for **Docker** (separate containers for API & dashboard)

---

## 🏗 Architecture

**High-level flow:**

```text
Raw Data (CSV)
      ↓
Preprocessing & Feature Engineering (src/data_preprocessing.py)
      ↓
Train & Evaluate Models (src/train.py)
      ↓
Log Experiments (MLflow) & Save Best Model (models/best_model.pkl)
      ↓
Serve Model via FastAPI (api/main.py)
      ↓
Consume API from Streamlit Dashboard (dashboard/app.py)

```
## 📂Project Structure
```
customer-churn-ml/
│
├── api/
│   └── main.py                # FastAPI app (prediction API)
│
├── dashboard/
│   └── app.py                 # Streamlit dashboard for UI & analytics
│
├── data/
│   ├── raw/                   # Raw dataset (Telco Customer Churn CSV)
│   └── processed/             # (Optional) processed data
│
├── models/
│   └── best_model.pkl         # Trained best model (sklearn pipeline)
│
├── notebooks/
│   └── 01_eda_and_baseline.ipynb  # EDA and baseline model notebook
│
├── src/
│   ├── config.py              # Paths & configuration
│   ├── data_preprocessing.py  # Data loading & preprocessing pipeline
│   ├── train.py               # Model training & MLflow logging
│   ├── inference.py           # Model loading & prediction helper
│   ├── schemas.py             # Pydantic schema for API input
│   └── __init__.py
│
├── docker/
│   ├── Dockerfile.api         # Dockerfile for FastAPI service
│   └── Dockerfile.streamlit   # Dockerfile for Streamlit dashboard
│
├── docker-compose.yml         # (Optional) Compose setup for API + UI
├── requirements.txt           # Project dependencies
├── .gitignore
└── README.md
```
## 📊 Dataset
Source: Telco Customer Churn dataset (IBM Sample / Kaggle)

⚙️ Setup & Installation (Local)
1️⃣ Clone the repository
```
git clone https://github.com/rhanand01/customer-churn-ml-pipeline.git
cd customer-churn-ml-pipeline
```
2️⃣ Create and activate virtual environment (optional but recommended)
```
python -m venv env
```
# Windows (PowerShell / CMD)
```
env\Scripts\activate
```
3️⃣ Install dependencies
```
pip install --upgrade pip
pip install -r requirements.txt
```
4️⃣ Place the dataset
Download the Telco Customer Churn CSV and place it in:
```
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv

```
🤖 Training the Model
Run the training script:
```
python -m src.train
```
Running the FastAPI Service
```
From project root: uvicorn api.main:app --reload
```
Running the Streamlit Dashboard
With FastAPI running in one terminal, open another terminal (same env activated):
```
streamlit run dashboard/app.py

```
API will be available at:
Docs (Swagger):
```
 http://127.0.0.1:8000/docs
````
Health check: 
```
http://127.0.0.1:8000/health
