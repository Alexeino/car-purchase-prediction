from fastapi import FastAPI
import joblib
from pydantic import BaseModel
from decimal import Decimal

app = FastAPI(title="Car Purchasing Prediction")

regressor = joblib.load("data/regressor.pkl")

class ModelInput(BaseModel):
    age: float
    annual_salary: float
    credit_card_debt: float
    net_worth: float

@app.get("/")
def index():
    return {"message": "Hi there !"}

@app.post("/predict")
def predict_purchasing_capacity(model_input: ModelInput):
    if regressor is None:
        return {"error": "Model is not loaded properly. Please check the pickle file."}
    # import pdb; pdb.set_trace()
    data = model_input.model_dump()
    age = data["age"]
    annual_salary = data["annual_salary"]
    credit_card_debt = data["credit_card_debt"]
    net_worth = data["net_worth"]
    
    prediction = regressor.predict([[age,annual_salary,credit_card_debt,net_worth]])
    return {'prediction':prediction[0]}