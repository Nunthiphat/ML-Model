from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# โหลดโมเดล
model = joblib.load("new_random_forest_model.pkl")

# สร้าง FastAPI app
app = FastAPI()

# สร้างโครงสร้างข้อมูลที่รับจากผู้ใช้
class HeartDiseaseInput(BaseModel):
    Age: int
    Sex: int
    ChestPainType: int
    RestingBP: int
    Cholesterol: int
    FastingBS: int
    RestingECG: int
    MaxHR: int
    ExerciseAngina: int
    Oldpeak: float
    ST_Slope: int

# API สำหรับทำนาย
@app.post("/predict/")
def predict(data: HeartDiseaseInput):
    # แปลงข้อมูลเป็น NumPy Array
    input_data = np.array([[data.Age, data.Sex, data.ChestPainType, data.RestingBP, 
                            data.Cholesterol, data.FastingBS, data.RestingECG, 
                            data.MaxHR, data.ExerciseAngina, data.Oldpeak, data.ST_Slope]])
    
    # ทำการทำนาย
    prediction = model.predict(input_data)
    
    return {"prediction": int(prediction[0])}
