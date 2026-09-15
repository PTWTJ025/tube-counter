import os
import sys
import uuid
import base64
import cv2
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import torch

app = FastAPI(title="Tube Counter AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = Path(__file__).parent / "model" / "best.pt"
_yolo_model: Optional[YOLO] = None

def get_model() -> YOLO:
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO(str(MODEL_PATH))
    return _yolo_model

def count_with_yolo(image_bytes: bytes, conf: float = 0.25, iou: float = 0.45) -> tuple:
    # 1. Decode image bytes for YOLO
    import numpy as np
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    # 2. Run inference
    model = get_model()
    results = model.predict(
        source=img,
        conf=conf,
        iou=iou,
        imgsz=640,
        device=0 if torch.cuda.is_available() else "cpu",
        verbose=False
    )
    result = results[0]
    count = len(result.boxes)
    
    # 3. Extract boxes
    boxes = []
    for i, box in enumerate(result.boxes):
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        boxes.append({
            "index": i + 1,
            "x1": round(x1), "y1": round(y1),
            "x2": round(x2), "y2": round(y2),
            "conf": round(float(box.conf[0]), 3)
        })

    # 4. Draw labels on the image
    annotated = result.plot(labels=False, conf=False)
    for box in boxes:
        idx = str(box["index"])
        x_center = int((box["x1"] + box["x2"]) / 2)
        y_center = int((box["y1"] + box["y2"]) / 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (tw, th), _ = cv2.getTextSize(idx, font, font_scale, thickness)
        tx = x_center - tw // 2
        ty = y_center + th // 2
        # Outline
        cv2.putText(annotated, idx, (tx, ty), font, font_scale, (0, 0, 0), thickness + 2)
        # White text
        cv2.putText(annotated, idx, (tx, ty), font, font_scale, (255, 255, 255), thickness)
        
    # 5. Encode annotated image to base64
    _, buffer = cv2.imencode('.jpg', annotated)
    b64_str = base64.b64encode(buffer).decode('utf-8')
    
    return count, boxes, b64_str

@app.get("/")
def root_status():
    return {
        "message": "Tube Counter AI API is running! 🚀",
        "frontend_url": "https://tube-counter-27b29.web.app",
        "status": "online"
    }

@app.get("/api/health")
def health_check():
    return {"status": "ok", "service": "AI Microservice"}

@app.post("/api/detect")
async def detect_tubes(
    file: UploadFile = File(...),
    conf: float = Form(0.25),
    iou: float = Form(0.45),
):
    try:
        content = await file.read()
        count, boxes, b64_str = count_with_yolo(content, conf=conf, iou=iou)
        
        return {
            "success": True,
            "count": count,
            "boxes": boxes,
            "annotated_image_base64": f"data:image/jpeg;base64,{b64_str}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
