from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="AICropSentinel API")


class DetectionResult(BaseModel):
    plant_id: str
    crop: str
    disease: str
    confidence: float
    spray_status: str = "pending"


@app.get("/")
def home():
    return {
        "status": "AICropSentinel API is running"
    }


@app.post("/api/rover/detection")
def receive_detection(data: DetectionResult):
    return {
        "status": "success",
        "message": "Detection received",
        "data": data
    }