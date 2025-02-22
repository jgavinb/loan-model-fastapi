# Loan Default Prediction API

A REST API for predicting loan default risk using a machine learning model trained with imbalanced data handling techniques. Built with FastAPI, imbalanced-learn, and Joblib.

## Features

- **Imbalanced Data Handling**: Utilizes various oversampling techniques from `imblearn` (e.g., SMOTE, ADASYN) to address class imbalance.
- **Model Persistence**: Trained model and scaler saved using `joblib` for efficient storage and loading.
- **REST API**: FastAPI-powered endpoint for real-time predictions.
- **Scalable Inputs**: Input data is scaled using a pre-trained scaler before prediction.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/loan-default-prediction.git
   cd loan-default-prediction
