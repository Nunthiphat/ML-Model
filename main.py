from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

# สร้าง FastAPI App
app = FastAPI()

# เปิดใช้งาน CORS (อนุญาตทุกโดเมนสำหรับพัฒนา)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # หรือเปลี่ยนเป็น ["http://localhost:3000"] เพื่อให้ปลอดภัยขึ้น
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# โหลดโมเดล Machine Learning
model = joblib.load("new_random_forest_model.pkl")

# สร้าง Model สำหรับรับข้อมูลจากผู้ใช้
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

# สร้าง API สำหรับพยากรณ์โรคหัวใจ
@app.post("/predict/")
async def predict(data: HeartDiseaseInput):
    # แปลงข้อมูลเป็น NumPy Array
    input_data = np.array([[data.Age, data.Sex, data.ChestPainType, data.RestingBP, 
                            data.Cholesterol, data.FastingBS, data.RestingECG, 
                            data.MaxHR, data.ExerciseAngina, data.Oldpeak, data.ST_Slope]])
    
    # ทำนายผล
    prediction = model.predict(input_data)
    
    return {"prediction": int(prediction[0])}

# สร้าง Endpoint ทดสอบ
@app.get("/test/")
async def test():
    return {"message": "Hello, API is working!"}

# รันเซิร์ฟเวอร์
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
