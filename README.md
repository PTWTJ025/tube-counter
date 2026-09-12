# Tube Counter — เว็บนับหลอดด้ายด้วยกล้อง

เครื่องมือสำหรับพนักงานหน้างานถ่ายรูปหลอดด้ายในตะกร้า แล้วระบบนับจำนวนอัตโนมัติด้วย
Computer Vision (Hough Circle Transform) พร้อมบันทึกประวัติแยกตามประเภทหลอด
(พันทับ / ใยแตก / อื่นๆ) ลงฐานข้อมูล

---

## 1. ภาพรวมสถาปัตยกรรม

```
┌─────────────────────────┐        ┌──────────────────────────┐
│   Browser (มือถือ/แท็บเล็ต)  │        │   FastAPI server          │
│                          │        │                            │
│  static/index.html       │  HTTP  │  main.py                  │
│  - เปิดกล้อง (getUserMedia)│ ─────▶ │  - เสิร์ฟหน้าเว็บ            │
│  - นับวงกลม (OpenCV.js)   │ ◀───── │  - รับ/บันทึกผลนับ (API)     │
│  - ส่งผลนับ + รูปย่อ       │  JSON  │  - อ่านประวัติ/ลบ/สรุปยอด    │
└─────────────────────────┘        │  - เขียนลง tube_counter.db │
                                    └──────────────┬─────────────┘
                                                    │
                                             ┌──────▼───────┐
                                             │ SQLite         │
                                             │ tube_counter.db│
                                             └───────────────┘
```

**แนวคิดสำคัญ:** การนับวงกลม (Hough Circle) รันที่ฝั่ง**เบราว์เซอร์**ทั้งหมด (ผ่าน OpenCV.js)
เซิร์ฟเวอร์มีหน้าที่แค่**เก็บผลลัพธ์** ไม่ต้องประมวลผลภาพซ้ำ ทำให้เซิร์ฟเวอร์เบาและรองรับ
หลายเครื่องพร้อมกันได้ง่าย

---

## 2. เทคโนโลยีที่ใช้

| ส่วน | เทคโนโลยี | เหตุผล |
|---|---|---|
| Frontend | HTML / CSS / Vanilla JS | ไฟล์เดียว รันได้ทุกเครื่องโดยไม่ต้อง build |
| Circle detection | OpenCV.js (client-side) | ไม่ต้องส่งรูปไป process ที่ server ทุกครั้ง |
| Backend | Python + FastAPI | เขียนน้อย เร็ว มี docs อัตโนมัติที่ `/docs` |
| Database | SQLite (ผ่าน SQLModel) | ไฟล์เดียว ไม่ต้องติดตั้ง DB server แยก |
| Dev server | Uvicorn | รองรับ auto-reload ตอนพัฒนา |

---

## 3. โครงสร้างโปรเจกต์

```
tube-counter/
├── venv/                  # virtual environment (ไม่ commit เข้า git)
├── static/
│   └── index.html         # หน้าเว็บนับหลอด
├── main.py                # FastAPI app + API endpoints
├── requirements.txt       # รายการ dependency
├── tube_counter.db        # SQLite (สร้างอัตโนมัติตอนรันครั้งแรก)
└── README.md              # เอกสารนี้
```

---

## 4. การติดตั้งและรันโปรเจกต์

### ขั้นตอนติดตั้งครั้งแรก

```bash
git clone <repo-url> tube-counter
cd tube-counter

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### รันตอนพัฒนา (auto-reload เมื่อแก้โค้ด)

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

เปิดเบราว์เซอร์ไปที่ `http://localhost:8000`
ถ้าจะทดสอบผ่านมือถือในวงแลนเดียวกัน ใช้ `http://<ip-เครื่อง-server>:8000`
(กล้องผ่านเว็บต้องใช้ HTTPS หรือ `localhost` เท่านั้น — ดูหัวข้อ 7)

### รันตอนใช้งานจริง (production)

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 2
```

---

## 5. แผน API Endpoints

| Method | Path | หน้าที่ |
|---|---|---|
| `GET` | `/` | เสิร์ฟหน้าเว็บ `static/index.html` |
| `POST` | `/api/counts` | บันทึกผลนับ 1 รายการ (ประเภท, จำนวน, รูปย่อ, เวลา) |
| `GET` | `/api/counts` | ดึงประวัติทั้งหมด (เรียงล่าสุดก่อน) |
| `DELETE` | `/api/counts/{id}` | ลบรายการที่บันทึกผิด |
| `GET` | `/api/counts/export` | ดาวน์โหลด CSV ของประวัติทั้งหมด |
| `GET` | `/api/counts/summary` | ยอดรวมจำนวนหลอด แยกตามประเภท |

### ตัวอย่าง request/response — บันทึกผลนับ

```http
POST /api/counts
Content-Type: application/json

{
  "type": "พันทับ",
  "count": 55,
  "thumbnail": "data:image/jpeg;base64,...",
  "captured_at": "2026-09-12T14:32:00+07:00"
}
```

```json
{
  "id": 17,
  "type": "พันทับ",
  "count": 55,
  "captured_at": "2026-09-12T14:32:00+07:00"
}
```

---

## 6. โครงสร้างฐานข้อมูล (ตาราง `tube_count`)

| คอลัมน์ | ชนิดข้อมูล | คำอธิบาย |
|---|---|---|
| `id` | INTEGER, primary key | รหัสรายการ |
| `type` | TEXT | ประเภทหลอด (พันทับ / ใยแตก / อื่นๆ) |
| `count` | INTEGER | จำนวนที่นับได้ (หลังแก้ไขโดยผู้ใช้) |
| `thumbnail` | TEXT | รูปย่อแบบ base64 (หรือ path ถ้าเก็บเป็นไฟล์แยก) |
| `captured_at` | DATETIME | เวลาที่ถ่ายรูป/บันทึก |

---

## 7. ข้อควรระวัง

- **กล้องผ่านเบราว์เซอร์ต้องการ HTTPS** (ยกเว้นเข้าผ่าน `localhost`) — ถ้าจะใช้งานจริงผ่านวง
  แลนโรงงานด้วย IP เครื่อง ต้องตั้ง reverse proxy (เช่น Nginx) พร้อมใบรับรอง SSL หรือใช้
  self-signed certificate สำหรับทดสอบภายใน
- **OpenCV.js โหลดจาก CDN** — ถ้าเครื่องที่ใช้งานไม่มีอินเทอร์เน็ต (LAN ปิด) ต้องดาวน์โหลด
  `opencv.js` มาเก็บไว้ในโปรเจกต์แล้วอ้างอิง path ภายในแทน
- **ข้อมูลรูปย่อ (thumbnail)** ถ้าเก็บเป็น base64 ในฐานข้อมูลตรงๆ ไฟล์ `.db` จะโตเร็ว
  ถ้าใช้งานเยอะควรเก็บไฟล์รูปแยกไว้ในโฟลเดอร์ `static/uploads/` แล้วเก็บแค่ path ในฐานข้อมูลแทน

---

## 8. แผนพัฒนาต่อ (ถ้าต้องการ)

- เปลี่ยนจาก Hough Circle Transform → โมเดล YOLO ที่เทรนเองด้วยรูปจริงจากโรงงาน
  (แม่นยำขึ้นในกรณีหลอดซ้อนทับ/เอียง)
- เพิ่มระบบ login แยกผู้บันทึกแต่ละกะ/แผนก
- Export รายงานสรุปรายวัน/รายสัปดาห์เป็น Excel
