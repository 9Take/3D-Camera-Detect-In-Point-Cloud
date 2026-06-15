# 👋 ระบบ Vision 3 มิติสำหรับ Heat Exchanger

> 🌐 ภาษา: [English](README.md) | **ไทย**

> README นี้อธิบายภาพรวมการทำงานของทั้งระบบ แยกตามส่วนงาน เพื่อให้วิศวกรคนใหม่
> สามารถพัฒนาต่อได้ อ่านหน้านี้ก่อน แล้วค่อยเจาะไปยังส่วนที่ต้องการ

---

## ระบบนี้ทำอะไร (สรุปย่อหน้าเดียว)

PLC สั่งให้ระบบ "สแกนเดี๋ยวนี้ โดยใช้โมเดล/โปรแกรมหมายเลข N" จากนั้นเราดึงภาพ RGB-D
จากกล้อง Intel RealSense, ค้นหาฟีเจอร์ที่สอนไว้ล่วงหน้า (template) ในภาพสีด้วยการจับคู่
แบบ SIFT, อ่านค่าความลึก (depth) ที่จุดที่เจอ, ยกขึ้นเป็นตำแหน่ง 3 มิติพร้อมทิศทางของพื้นผิว,
แปลงจาก **เฟรมกล้อง (camera frame)** ไปเป็น **เฟรมฐานหุ่นยนต์ (robot base frame)** โดยใช้
ผลการคาลิเบรต hand-eye แล้วเขียนค่าตำแหน่ง 6-DOF (X, Y, Z, A, B, C) กลับไปยังรีจิสเตอร์ของ PLC
เพื่อให้แขนกลเคลื่อนที่ไปยังจุดนั้น การสื่อสารกับ PLC ใช้ Mitsubishi
**MC Protocol (Type3E binary)** ผ่าน TCP

```
PLC ──"สแกน, โปรแกรม N"──►  PC (โค้ดนี้)  ──X,Y,Z,A,B,C ต่อจุด──►  PLC ──► หุ่นยนต์
 ▲                              │
 └──── status/heartbeat ───  RealSense D435i (RGB + Depth)
```

---

## เอกสารส่งมอบงาน (อ่านตามลำดับนี้)

| # | เอกสาร | เนื้อหา |
|---|--------|---------|
| 1 | [getting-started.th.md](docs/handover/getting-started.th.md) | การติดตั้ง, การรัน, Docker, ปุ่มควบคุมโหมด debug, วิธีศึกษาโค้ด |
| 2 | [core.th.md](docs/handover/core.th.md) | สมองของระบบ vision: `core/detector.py` (หา 2D) + `core/transformer.py` (2D→3D pose) |
| 3 | [communication.th.md](docs/handover/communication.th.md) | `communication/realsense.py` (กล้อง) + `communication/plc_comm.py` (PLC I/O) + ตารางรีจิสเตอร์ |
| 4 | [calibration.th.md](docs/handover/calibration.th.md) | คาลิเบรต hand-eye: ทำไม, วิธีรัน `calibration/aruco_calibate.py`, วิธีตรวจสอบ |
| 5 | [tools.th.md](docs/handover/tools.th.md) | สคริปต์ช่วยเหลือ: สร้าง template, ถอดรหัสคำของ PLC, ตรวจจับ board, geometry, solver แบบ offline |
| 6 | [configuration.th.md](docs/handover/configuration.th.md) | อธิบายทุก key ใน `config.yaml` |
| 7 | [main-loop.th.md](docs/handover/main-loop.th.md) | `main.py` ร้อยทุกอย่างเข้าด้วยกันอย่างไร, หนึ่งรอบ trigger ทีละขั้น |

ยังมีเอกสารเชิงลึกที่อ้างอิงเลขบรรทัดของโค้ดอยู่ที่
[docs/methodology.th.md](docs/methodology.th.md) — ใช้เมื่อต้องการเลขบรรทัดที่แม่นยำ

---

## แผนผังโปรเจกต์ (อะไรอยู่ที่ไหน)

```
3D-Camera-Detect-In-Point-Cloud/
├── main.py                 ← จุดเริ่มทำงานตอนรันจริง (production) ไฟล์นี้คือตัวที่รันใช้งานจริง
├── config.yaml             ← การตั้งค่าทั้งหมด (กล้อง, รีจิสเตอร์ PLC, โปรแกรม, คาลิเบรต)
├── requirements.txt        ← ไลบรารี Python (Python 3.8)
├── Dockerfile / docker-compose.yml
│
├── core/
│   ├── detector.py         ← จับคู่ template ด้วย SIFT → พิกเซล 2D + ค่าความมั่นใจ (confidence)
│   └── transformer.py      ← พิกเซล + depth → จุด 3D + ทิศทาง 6-DOF
│
├── communication/
│   ├── realsense.py        ← ตัวห่อหุ้มกล้อง RealSense (RGB-D ที่ align แล้ว, ฟิลเตอร์ depth)
│   └── plc_comm.py         ← อ่าน/เขียน MC-protocol, reconnect, heartbeat, เขียน slot
│
├── calibration/
│   ├── aruco_calibate.py   ← เก็บข้อมูล + แก้สมการคาลิเบรต hand-eye (รันเป็นครั้งคราว)
│   ├── hand_eye_result.npz ← ผลคาลิเบรตที่บันทึกไว้ + ค่า pose ดิบ
│   └── capture_log.csv     ← log ดิบของแต่ละ pose (สำหรับ debug แบบ offline)
│
├── tools/
│   ├── create_template.py  ← สอน template ใหม่ (คลิกเลือกจุดบนภาพจากกล้อง)
│   ├── plc_decode.py        ← เข้ารหัส/ถอดรหัส int32 ↔ คำของ PLC (ตรรกะล้วน, เทสต์ได้)
│   ├── board_detect.py      ← หา pose ของ ChArUco board (ใช้ในการคาลิเบรต)
│   ├── geometry.py          ← คณิตศาสตร์ มุม KUKA ↔ เมทริกซ์ (numpy ล้วน)
│   └── cal_ressult_calib.py ← solver แบบ brute-force ออฟไลน์ เพื่อ debug คาลิเบรตที่ผิดพลาด
│
├── data/
│   ├── templates/          ← template ที่สอนไว้ จัดเป็น ProgramX/PointY/*.png + meta.json
│   └── logs/               ← position_mem.json (สแกนล่าสุด), current_detect.json
│
└── docs/                   ← methodology (อ้างอิงเชิงลึก) + handover/ (คู่มือแยกตามส่วน)
```

---

## วิธีศึกษาโปรเจกต์นี้เมื่อหาทางไม่เจอ

1. **เริ่มจาก data flow** ไม่ใช่จากไฟล์ อ่าน [main-loop.th.md](docs/handover/main-loop.th.md) — มันเดิน
   ผ่านหนึ่ง trigger ตั้งแต่บิตของ PLC จนถึงผลลัพธ์ที่ส่งกลับ PLC โมดูลอื่น ๆ ทั้งหมดต่อยอดจากตรงนี้
2. **รันในโหมด `--debug` โดยไม่ต้องมีหุ่นยนต์** สั่ง trigger จากคีย์บอร์ดได้
   และดูหน้าต่าง live + 3D viewer ได้ ดู [getting-started.th.md](docs/handover/getting-started.th.md)
3. **ฝั่ง PLC ก็แค่รีจิสเตอร์** ข้อตกลงทั้งหมดกับ PLC อยู่ในส่วน
   `plc:` ของ `config.yaml` ถ้ามีอาการ "คุยกับ PLC ไม่ได้" ตารางนี้คือที่แรกที่ต้องดู
   — ดู [communication.th.md](docs/handover/communication.th.md)
4. **ใช้ log** `data/logs/position_mem.json` คือผลลัพธ์เต็มของการสแกนครั้งล่าสุด
   `calibration/capture_log.csv` คือทุก pose ของการคาลิเบรต ทั้งสองไฟล์ทำมาเพื่อ debug แบบ offline
5. **โฟลเดอร์ `memory/`** บันทึกข้อควรระวังที่ไม่ชัดเจน ซึ่งได้มาจากการเจอปัญหาจริง
