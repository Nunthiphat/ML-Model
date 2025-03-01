from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel

# โหลดโมเดลที่เทรนไว้
model = joblib.load("new_random_forest_model.pkl")

# สร้าง API ด้วย FastAPI
app = FastAPI()

# กำหนดโครงสร้างข้อมูลที่รับเข้ามา
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

@app.post("/predict/")
def predict(data: HeartDiseaseInput):
    # แปลง input เป็น numpy array
    input_data = np.array([[data.Age, data.Sex, data.ChestPainType, data.RestingBP, 
                            data.Cholesterol, data.FastingBS, data.RestingECG, data.MaxHR, 
                            data.ExerciseAngina, data.Oldpeak, data.ST_Slope]])

    # ทำการทำนาย
    prediction = model.predict(input_data)[0]
    
    # ส่งผลลัพธ์กลับไป
    return {"prediction": int(prediction)}
