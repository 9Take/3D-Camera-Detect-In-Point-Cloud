# 1. เริ่มต้นใช้งาน (Getting Started)

> 🌐 ภาษา: [English](getting-started.md) | **ไทย**

วิธีติดตั้ง รัน และพัฒนาระบบ เริ่มที่นี่ในวันแรก

---

## ฮาร์ดแวร์ที่ต้องมี

- **Intel RealSense D435i** (หรือ D4xx รุ่นใด ๆ) เสียบกับพอร์ต USB 3.0
- **Mitsubishi PLC** ที่เข้าถึงได้ในเครือข่าย พูดภาษา **MC Protocol Type3E (binary)**
  ที่อยู่เริ่มต้นคือ `192.168.1.165:5010` (ตั้งใน `config.yaml`)
- **หุ่นยนต์** (KUKA) — แต่ **ไม่จำเป็น** ต้องมีตอนพัฒนาฝั่ง vision ใช้โหมด `--debug`
  กับตัวจำลอง PLC หรือใช้แค่กล้องก็ได้

งานพัฒนาส่วนใหญ่ทำได้ด้วย **กล้องอย่างเดียว** (สร้าง template, จูนการตรวจจับ) และ
**PLC อย่างเดียว** (ทดสอบ handshake) ส่วนลูปเต็มรูปแบบต้องใช้ครบทั้งสามอย่าง

---

## การติดตั้ง (แบบ native แนะนำสำหรับการพัฒนา)

ต้องใช้ Python **3.8** (wheel ของ RealSense + Open3D ผูกกับเวอร์ชันนี้)

```bash
cd "3D-Camera-Detect-In-Point-Cloud"
python3.8 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

ไลบรารีสำคัญและเหตุผล:
- `pyrealsense2` — ไดรเวอร์กล้อง
- `opencv-python==4.7.0.72` — SIFT, homography, ArUco **ล็อกเวอร์ชันไว้** (ดูหมายเหตุบั๊ก
  ในหน้าคาลิเบรต) อย่าอัปเกรดพร่ำเพรื่อ
- `open3d==0.17.0` — point cloud + การประมาณ normal + 3D viewer (บน ARM/Jetson การ build
  ผ่าน Docker จะลดเป็น 0.16.0 เพราะไม่มี wheel 0.17 สำหรับ aarch64 — API ที่ใช้เข้ากันได้)
- `pymcprotocol` — ไคลเอนต์ MC protocol ของ Mitsubishi
- `scipy` — คณิตศาสตร์การหมุน/Euler
- `numpy==1.24.3`, `PyYAML`

---

## การรัน

```bash
# โหมด production: รอ PLC ส่งหมายเลขโปรแกรม + trigger
python main.py

# โหมดพัฒนาโดยไม่ต้องมีหุ่นยนต์: trigger จากคีย์บอร์ด + 3D viewer หลังสแกนแต่ละครั้ง
python main.py --debug
```

`main.py` คือจุดเริ่มทำงานตอนรันจริง **เพียงจุดเดียว**

### สิ่งที่จะเห็น
- หน้าต่าง **"Vision System - Live"**: ภาพสดจากกล้อง พร้อมวาดเป้าหมายที่ดีที่สุดต่อจุด
  และ heartbeat บรรทัดเดียวในคอนโซล
- หลัง trigger จะมี **"Trigger Result"** เป็น grid: หนึ่ง tile ต่อหนึ่ง sub-template
  โดย template ที่ถูกเลือกเป็น best ต่อจุดจะถูกตีกรอบสีเขียว

---

## ปุ่มควบคุมคีย์บอร์ด (ต้องโฟกัสที่หน้าต่าง live)

ใช้ได้เสมอ:

| ปุ่ม | การทำงาน |
|------|----------|
| `p` | เปิด 3D point-cloud viewer ของการสแกน **ครั้งล่าสุด** |
| `q` / `ESC` | ออกจากโปรแกรม |

เฉพาะใน `--debug`:

| ปุ่ม | การทำงาน |
|------|----------|
| `1`–`9` | เลือกหมายเลขโปรแกรมด้วยมือ |
| `t` | สั่ง trigger เอง (สแกนเดี๋ยวนี้) |
| `b` | สลับเข้า/ออก **โหมดย่อย PLC-test** |

ใน **โหมดย่อย PLC-test** (ให้วิศวกร PLC ตรวจสอบฝั่งของตนเองในการทำ handshake):

| ปุ่ม | การทำงาน |
|------|----------|
| `1`–`9` | เขียนหมายเลขโปรแกรมนั้นไปที่ `program_no_test_device` (D1500) |
| `t` | ส่งพัลส์ที่ `trigger_test_device` |
| `b` | ออกจากโหมดย่อย PLC-test |

---

## การรันด้วย Docker

`docker-compose.yml` ตั้งค่าให้ส่งผ่านกล้อง + display ไว้แล้ว:

```bash
xhost +local:                 # อนุญาตให้ container เปิดหน้าต่างบน X server ของคุณ
docker compose up --build
```

มันจะรัน `python3 main.py`, ใช้ `network_mode: host` (เพื่อให้เข้าถึง PLC ได้),
`privileged` + `/dev/bus/usb` (เข้าถึง RealSense) และ mount `./data` กับ `./config.yaml`
เพื่อให้แก้ template/config ได้สด ๆ โดยไม่ต้อง build ใหม่

---
