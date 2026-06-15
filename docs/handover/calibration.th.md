# 4. Calibration — Hand-Eye

> 🌐 ภาษา: [English](calibration.md) | **ไทย**

นี่คือส่วนที่เกิดข้อผิดพลาดได้ง่ายที่สุดของทั้งระบบ อ่านหน้านี้ให้จบก่อนรัน

**ไฟล์:** [`calibration/aruco_calibate.py`](../../calibration/aruco_calibate.py)
**ตัวช่วย:** [`tools/board_detect.py`](../../tools/board_detect.py),
[`tools/geometry.py`](../../tools/geometry.py),
[`tools/plc_decode.py`](../../tools/plc_decode.py)

---

## ทำไมต้องคาลิเบรต

กล้องติดอยู่บนข้อมือหุ่นยนต์ ("eye-in-hand") การจะสั่งหุ่นยนต์ไปยังจุดที่เรา *เห็น* เราต้องรู้
การแปลงคงที่จาก **เฟรมกล้อง** ไปยัง **เฟรม gripper/TCP** — คือคู่ `R_cam2gripper`, `t_cam2gripper`
นั่นคือผลของการคาลิเบรต hand-eye เมื่อรวมกับ pose gripper→base ที่รู้ค่าของหุ่นยนต์ จะทำให้
แปลงจุดใด ๆ ในเฟรมกล้องไปเป็นพิกัดฐานหุ่นยนต์ได้

ผลลัพธ์อยู่ใน `config.yaml` ใต้ `robot.hand_eye_rotation` / `robot.hand_eye_translation`
และ `build_cam2base()` ใน `main.py` จะรวมมันกับ pose ถ่ายภาพเพื่อได้เมทริกซ์ `H_cam2base` 4×4 เต็ม

---

## วิธีการแบบภาพเดียวจบ

เราวาง **ChArUco board** (กระดานหมากรุกที่มี ArUco marker) ตรึงไว้ในโลก แล้วขยับหุ่นยนต์ไปยัง
~16-20 pose ที่แตกต่างกัน ที่แต่ละ pose เราบันทึกสองอย่าง:
- **pose หุ่นยนต์** (gripper→base) อ่านจาก PLC
- **pose ของ board ในกล้อง** (target→cam) จาก `detect_board_pose`

`cv2.calibrateHandEye(...)` ของ OpenCV แก้สมการหาการแปลง camera→gripper จากคู่เหล่านั้น
(เป็นปัญหา `AX = XB` คลาสสิก)

```
สำหรับแต่ละ pose จาก N pose:
  หุ่นยนต์อยู่ที่ pose i ──► PLC ยก M2000 ──► PC เห็น board, อ่าน pose หุ่นยนต์ ──► PC ยก M2001
                                                                              ──► PLC ขยับไป pose ถัดไป
รวบรวม N คู่ → cv2.calibrateHandEye(method=PARK) → R/t cam→gripper
```

## Handshake 4 เฟส กับ PLC

การเก็บข้อมูลถูกขับเคลื่อนโดยหุ่นยนต์ และ interlock ไว้ เพื่อให้ pose ไม่มีทางถูกบันทึกระหว่างเคลื่อนที่
หรือถูกพลาด:

```
state: wait_trigger
  PLC ตั้ง M2000=1 (KUKA ไปถึง pose)
    → PC รอ SETTLE_SEC ให้แขนนิ่ง + คำ pose เสถียร
    → PC อ่าน pose ของ board + pose หุ่นยนต์ (สองครั้ง ต้องตรงกันภายใน POSE_STABLE_TOL)
    → PC บันทึกคู่นั้น, ตั้ง M2001=1 ("กล้องเสร็จเรียบร้อย")
state: wait_release
  PLC เห็น M2001=1, ลด M2000=0
    → PC เห็น M2000=0, ลด M2001=0 → กลับไป wait_trigger สำหรับ pose ถัดไป
```

`M2001` ถูก **ค้างไว้สูง** จนกว่า PLC จะลด `M2000` ดังนั้น ack จึงพลาดไม่ได้

### กลไกป้องกันที่มีอยู่ (ทำไมแต่ละรอบลูปถึง "ข้าม" ได้)
- **pose เป็นศูนย์ทั้งหมด** → PLC ยก trigger ก่อนเขียน `D2000` ให้รอ อย่าบันทึก
- **pose เปลี่ยนระหว่างสองครั้งที่อ่าน** → PLC กำลังเขียน/desync ข้ามรอบนี้
- **pose กล้องเป็น NaN** → ตรวจจับไม่ดี ลองเฟรมใหม่
- **board reproj > `max_reproj_px`** → board กำกวม/พลิก ปฏิเสธ (ใน `board_detect`)
- **board Z อยู่นอก 50–3000 mm** → ไม่ได้อยู่หน้ากล้องจริง ๆ ปฏิเสธ

---

## วิธีรัน

1. **พิมพ์/ติดตั้ง ChArUco board** เรขาคณิตของมันต้องตรงกับ `config.yaml → charuco:`
   (`squares_x`, `squares_y`, `square_length`, `marker_length`, `dictionary`) ถ้าตั้งผิด
   ทุก pose จะผิดหมด
2. **ตั้งโปรแกรมหุ่นยนต์** ให้ไล่ผ่าน ~16 pose ที่หลากหลาย โดยยก `M2000` ที่แต่ละ pose แล้ว
   รอ `M2001` **เปลี่ยนทิศทาง (orientation) ให้มาก ๆ** (หมุนรอบ ≥2 แกน) — ถ้า translate อย่างเดียว
   จะได้คำตอบที่ degenerate และใช้ไม่ได้
3. **จูน `calibration:` ใน config** ถ้าจำเป็น (`total_poses`, `min_charuco_corners`,
   `max_reproj_px`, `settle_sec`, `pose_stable_tol`)
4. รัน:
   ```bash
   python calibration/aruco_calibate.py
   ```
   ดูที่หน้าต่าง live: มันแสดงจำนวนมุม, บิต handshake ทั้งสอง, และกำลังรออะไรอยู่ กด `q` เพื่อยกเลิก
5. หลังเก็บครบ `total_poses` มันจะแก้สมการ, พิมพ์ทั้ง 5 วิธีเพื่อเปรียบเทียบ, และบันทึก:
   - `calibration/hand_eye_result.npz` — R/t + **เมทริกซ์ดิบของทุก pose**
   - `calibration/capture_log.csv` — pose KUKA ดิบ, คำ PLC ดิบ, reproj error ต่อ pose

---

## รู้ได้อย่างไรว่าสำเร็จ (การตรวจสอบ)

สคริปต์จะพิมพ์ **residual**: เนื่องจาก board ตรึงอยู่ในโลก ตำแหน่งที่คำนวณได้ของมันใน
**เฟรม base** ต้องเป็นจุดเดียวกันทุก pose การกระจายของจุดเหล่านั้น = error ของการแก้สมการ

- **RMS residual < ~50 mm** → สอดคล้อง ใช้ได้ (เราเคยทำได้ ~29 mm)
- **> 50 mm** → ไม่น่าเชื่อถือ อาจเป็นเพราะ pose degenerate (เปลี่ยน A/B/C ให้มากขึ้น) หรือ
  หน่วย/ทิศทางการแปลงไม่ตรงกันใน pose หุ่นยนต์

ถ้าผลแย่ **อย่าเดา** — ใช้ตัว debug แบบ offline
[`tools/cal_ressult_calib.py`](../../tools/cal_ressult_calib.py) มันโหลด CSV+NPZ กลับมา และ
brute-force ทั้ง 5 วิธี × 32 convention/sign ของมุม พร้อม diagnostic ความสอดคล้องของคู่ ที่ flag
pose แย่ ๆ ทีละตัวให้ทิ้ง ดู [tools.th.md](tools.th.md)

---

## นำผลไปใช้งานจริง (production)

`aruco_calibate.py` บันทึกเป็น `.npz` แต่ `main.py` อ่านผลคาลิเบรตจาก **`config.yaml`**
(`robot.hand_eye_rotation` / `robot.hand_eye_translation`) หลังคาลิเบรตสำเร็จ ให้คัดลอก
`R_cam2gripper` (3×3) และ `t_cam2gripper` (เมตร) ของ PARK ไปใส่ใน key เหล่านั้น นั่นคือขั้นตอน
ส่งต่อด้วยมือ — ไม่มีการเขียนอัตโนมัติ

> **หน่วย:** การคาลิเบรตป้อน translation ของหุ่นยนต์เป็น **mm** ดังนั้น `t_cam2gripper` ที่
> พิมพ์/บันทึกจึงเป็น **mm** ส่วน `config.yaml → robot.hand_eye_translation` เป็น **เมตร**
> ต้องหารด้วย 1000 ตอนคัดลอก (เช่น `-44.9 mm → -0.044947`) ส่วน `R_cam2gripper` ไม่มีหน่วย
> คัดลอกได้ตรง ๆ
