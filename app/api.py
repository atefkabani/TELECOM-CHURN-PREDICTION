from flask import Flask, request, jsonify , render_template
import joblib
import pandas as pd
from pathlib import Path

app = Flask(__name__)

model_path = Path(__file__).resolve().parent.parent.joinpath('model/log_reg_pipeline.joblib')

def load_model():
    model = joblib.load(model_path)
    return model

def extract_data(data):
    gender = data.get('gender')
    SeniorCitizen = data.get('SeniorCitizen')
    Partner = data.get('Partner')
    Dependents = data.get('Dependents')
    tenure = data.get('tenure')
    PhoneService = data.get('PhoneService')
    MultipleLines = data.get('MultipleLines')
    InternetService = data.get('InternetService')
    OnlineSecurity = data.get('OnlineSecurity')
    OnlineBackup = data.get('OnlineBackup')
    DeviceProtection = data.get('DeviceProtection')
    TechSupport = data.get('TechSupport')
    StreamingTV = data.get('StreamingTV')
    StreamingMovies = data.get('StreamingMovies')
    Contract = data.get('Contract')
    PaperlessBilling = data.get('PaperlessBilling')
    PaymentMethod = data.get('PaymentMethod')
    MonthlyCharges = data.get('MonthlyCharges')
    TotalCharges = data.get('TotalCharges')

    data_df = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [SeniorCitizen],
        "Partner": [Partner],
        "Dependents": [Dependents],
        "tenure": [int(tenure)],
        "PhoneService": [PhoneService],
        "MultipleLines": [MultipleLines],
        "InternetService": [InternetService],
        "OnlineSecurity": [OnlineSecurity],
        "OnlineBackup": [OnlineBackup],
        "DeviceProtection": [DeviceProtection],
        "TechSupport": [TechSupport],
        "StreamingTV": [StreamingTV],
        "StreamingMovies": [StreamingMovies],
        "Contract": [Contract],
        "PaperlessBilling": [PaperlessBilling],
        "PaymentMethod": [PaymentMethod],
        "MonthlyCharges": [float(MonthlyCharges)],
        "TotalCharges": [float(TotalCharges)]
    })

    return data_df


@app.post('/predict')
def show():
    data = request.args
    if not data:
        return jsonify({"error": "No input data provided"}), 400
    if not all(key in data for key in ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges"]):
        return jsonify({"error": "Missing required input data"}), 400
    
    
    
    df = extract_data(data)
    pred = load_model().predict(df)[0]
    if pred == 1:
      return jsonify({"prediction": "Churn"})
    else:
      return jsonify({"prediction": "Not Churn"})
  
@app.get('/')
def index():
    return  render_template('index.html')
    
app.run()