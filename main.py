import os
import io
import sys
import json
import csv
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# -------------------------------------------------------------
# Computer Vision: Circle Counting Logic (OpenCV)
# -------------------------------------------------------------
def count_circles(image_path: str, out_path: Optional[str] = None, dp: float = 1.2,
                  min_dist: float = 25.0, param1: float = 80.0, param2: float = 35.0,
                  min_r: int = 18, max_r: int = 55, crop: Optional[Tuple[int, int, int, int]] = None):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    if crop is not None:
        y1, y2, x1, x2 = crop
        img = img[y1:y2, x1:x2]

    # resize down for speed + more stable Hough params (keep aspect ratio)
    h, w = img.shape[:2]
    scale = 1000 / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    gray = cv2.equalizeHist(gray)  # boost contrast between metal rim and dark hole

    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT,
        dp=dp, minDist=min_dist,
        param1=param1, param2=param2,
        minRadius=min_r, maxRadius=max_r
    )

    out = img.copy()
    count = 0
    circle_items = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        count = circles.shape[1]
        for i, (x, y, r) in enumerate(circles[0, :]):
            circle_items.append({"index": i + 1, "x": int(x), "y": int(y), "r": int(r)})
            cv2.circle(out, (x, y), r, (0, 255, 0), 2)
            cv2.circle(out, (x, y), 2, (0, 0, 255), 3)
            cv2.putText(out, str(i + 1), (x - 10, y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    cv2.putText(out, f"Count: {count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)

    if out_path:
        cv2.imwrite(out_path, out)

    return count, circle_items, out

# -------------------------------------------------------------
# JSON File Storage
# -------------------------------------------------------------
DATA_FILE = Path("records.json")

class TubeRecordModel(BaseModel):
    id: Optional[str] = None
    time_str: Optional[str] = ""
    tube_type: str = "ใยพันทับ"
    count: int = 0
    image_url: Optional[str] = None
    thumb_url: Optional[str] = None
    notes: Optional[str] = ""
    created_at: Optional[str] = None

def load_records() -> List[Dict[str, Any]]:
    if not DATA_FILE.exists():
        return []
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_records(records: List[Dict[str, Any]]) -> None:
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

# -------------------------------------------------------------
# FastAPI Application
# -------------------------------------------------------------
app = FastAPI(
    title="Tube Counter API",
    description="ระบบตรวจจับและนับหลอดด้าย (OpenCV & JSON Storage)",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
DIST_DIR = Path("dist")
STATIC_DIR = Path("static")
UPLOAD_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
if (DIST_DIR / "assets").exists():
    app.mount("/assets", StaticFiles(directory="dist/assets"), name="assets")

# -------------------------------------------------------------
# Web Page Route (Fallback when accessed directly on FastAPI)
# -------------------------------------------------------------
@app.get("/", response_class=FileResponse)
def serve_index():
    if (DIST_DIR / "index.html").exists():
        return FileResponse(str(DIST_DIR / "index.html"))
    if (STATIC_DIR / "index.html").exists():
        return FileResponse(str(STATIC_DIR / "index.html"))
    return FileResponse("index.html")

@app.get("/api/health")
def health_check():
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}

# -------------------------------------------------------------
# API Endpoints
# -------------------------------------------------------------
@app.get("/api/records")
def get_records():
    records = load_records()
    # Sort descending by id or created_at
    return list(reversed(records))

@app.post("/api/records")
def add_record(record: TubeRecordModel):
    records = load_records()
    rec_dict = record.model_dump()
    if not rec_dict.get("id"):
        rec_dict["id"] = f"{int(datetime.now().timestamp() * 1000)}"
    if not rec_dict.get("time_str"):
        rec_dict["time_str"] = datetime.now().strftime("%d/%m/%Y, %H:%M")
    if not rec_dict.get("created_at"):
        rec_dict["created_at"] = datetime.now(timezone.utc).isoformat()
    
    records.append(rec_dict)
    save_records(records)
    return rec_dict

@app.delete("/api/records/{record_id}")
def delete_record(record_id: str):
    records = load_records()
    initial_len = len(records)
    records = [r for r in records if str(r.get("id")) != str(record_id)]
    if len(records) == initial_len:
        raise HTTPException(status_code=404, detail="Record not found")
    save_records(records)
    return {"message": f"Deleted record {record_id}", "id": record_id}

@app.post("/api/detect")
async def detect_tubes(
    file: UploadFile = File(...),
    dp: float = Form(1.2),
    min_dist: float = Form(25.0),
    param1: float = Form(80.0),
    param2: float = Form(35.0),
    min_r: int = Form(18),
    max_r: int = Form(55)
):
    ext = Path(file.filename or "image.jpg").suffix.lower()
    if ext not in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
        ext = ".jpg"
    
    unique_id = uuid.uuid4().hex
    raw_path = UPLOAD_DIR / f"raw_{unique_id}{ext}"
    annotated_path = UPLOAD_DIR / f"annotated_{unique_id}{ext}"

    content = await file.read()
    with open(raw_path, "wb") as f:
        f.write(content)

    try:
        count, circles, _ = count_circles(
            str(raw_path),
            out_path=str(annotated_path),
            dp=dp,
            min_dist=min_dist,
            param1=param1,
            param2=param2,
            min_r=min_r,
            max_r=max_r
        )
        return {
            "success": True,
            "count": count,
            "circles": circles,
            "raw_image_url": f"/uploads/{raw_path.name}",
            "annotated_image_url": f"/uploads/{annotated_path.name}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    ext = Path(file.filename or "image.jpg").suffix.lower()
    if ext not in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
        ext = ".jpg"
    unique_filename = f"img_{uuid.uuid4().hex}{ext}"
    target_path = UPLOAD_DIR / unique_filename
    
    content = await file.read()
    with open(target_path, "wb") as f:
        f.write(content)
        
    return {
        "url": f"/uploads/{unique_filename}",
        "filename": unique_filename,
        "size": len(content)
    }

@app.get("/api/export")
def export_csv():
    records = load_records()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "เวลา", "ประเภท", "จำนวน", "หมายเหตุ", "รูปรวม"])
    
    for r in reversed(records):
        writer.writerow([
            r.get("id", ""),
            r.get("time_str", ""),
            r.get("tube_type", ""),
            r.get("count", 0),
            r.get("notes", ""),
            r.get("image_url") or r.get("thumb_url") or ""
        ])
    
    output.seek(0)
    return Response(
        content=output.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename=tube-count-{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"}
    )

@app.get("/api/export/json")
def export_json():
    records = load_records()
    content = json.dumps(records, ensure_ascii=False, indent=2)
    return Response(
        content=content,
        media_type="application/json; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename=tube-records-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"}
    )

# -------------------------------------------------------------
# CLI Entry Point
# -------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) > 2:
        src = sys.argv[1]
        dst = sys.argv[2]
        kwargs = {}
        if len(sys.argv) > 6:
            y1, y2, x1, x2 = map(int, sys.argv[3:7])
            kwargs["crop"] = (y1, y2, x1, x2)
        n, _, _ = count_circles(src, dst, **kwargs)
        print(f"{src} -> detected {n} circles, saved to {dst}")
    else:
        import uvicorn
        port = int(os.environ.get("PORT", 8000))
        uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)