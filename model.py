# model.py
import joblib

model = joblib.load("rf_model.pkl")

def predict(features):
    return model.predict(features)
