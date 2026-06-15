# 7. Main Loop — ทุกอย่างประกอบกันอย่างไร

> 🌐 ภาษา: [English](main-loop.md) | **ไทย**

[`main.py`](../../main.py) คือตัวประสานงาน ถ้าเข้าใจไฟล์นี้ คุณก็เข้าใจทั้งระบบ ส่วนที่เหลือ
คือโมดูลที่มันเรียกใช้ อ่านหน้านี้ควบคู่กับเปิด `main.py` ไว้ข้าง ๆ

---

## ตอนเริ่มระบบ (`main()` → `setup_systems()`)

1. `load_config()` อ่าน `config.yaml`
2. `setup_systems()`:
   - เปิด `DepthCamera`
   - `load_detectors()` — สร้าง `ObjectDetector` หนึ่งตัวต่อโปรแกรม (โหลด template ล่วงหน้า)
   - สร้าง `PointCloudTransformer`
   - เชื่อมต่อ `PLCCommunicator` และเริ่ม **thread heartbeat**
   - ส่งสถานะ PLC เริ่มต้น: `error=ok`, `ready=1, busy=0, complete=0, error=0`
3. สร้าง `H_CAM2BASE` เริ่มต้นจาก `build_cam2base(robot_cfg, scan_pose)` — การแปลง camera→base
   โดยใช้ scan pose จาก *config* (จะ refresh สดทุก trigger ทีหลัง)

---

## ลูปไม่รู้จบ (ต่อหนึ่งรอบ)

```
ดึงเฟรม RGB-D
  └─ (จำกัดที่ poll_interval_sec) อ่าน program_no จาก PLC → ตั้ง current_program_no
render_live_view()  ── หน้าต่าง live: การตรวจจับที่ดีที่สุดต่อจุด + คำใบ้ปุ่ม + heartbeat คอนโซล
จัดการคีย์บอร์ด     ── q/ESC ออก · p ดู 3D · (debug) 1-9 โปรแกรม · t trigger · b PLC-test
  └─ (จำกัด) อ่านบิต trigger
ถ้าไม่ trigger: continue
```

เมื่อ trigger เกิด (บิต PLC = 1 หรือ `t` ในโหมด debug) → รัน **หนึ่งรอบสแกน**

---

## หนึ่งรอบสแกน (ส่วนสำคัญ)

```
1. set_status(ready=0, busy=1, complete=0, error=0); error_code = ok
2. ตรวจสอบ program_no  ── ถ้าไม่ถูกต้อง → error_code=invalid_program, error=1,
                          รอ trigger เคลียร์, ยกเลิกรอบ
3. read_robot_scan_pose(plc) จาก D2000  ── pose หุ่นยนต์สด
     └─ ถ้าใช้ได้: สร้าง H_CAM2BASE ใหม่จากมัน
     └─ ถ้าศูนย์ทั้งหมด/fail: ใช้ H_CAM2BASE จาก config (พร้อมเตือน)
4. จับเฟรมสแกนใหม่  ── ถ้ากล้อง fail → error_code=camera, ยกเลิกรอบ
5. detector.detect(scan_frame)  ── template ทั้งหมดที่เจอ
6. best_per_point(...)          ── เก็บเฉพาะ template ที่ confidence สูงสุดต่อจุด
7. แสดง grid "Trigger Result" (best per point ตีกรอบเขียว)
8. ถ้าไม่มีเป้าหมาย → report_no_results() (amount=0, error=no_targets, พัลส์ complete) → รอบถัดไป
9. transformer.extract_3d_data(พิกเซล best)  ── 6-DOF ต่อเป้าหมายในเฟรม point-cloud
     └─ ถ้ายกขึ้นไม่ได้เลย (depth ศูนย์ทั้งหมด) → report_no_results() → รอบถัดไป
10. เขียน amount_device = จำนวนเป้าหมาย (จำกัดที่ max_points)
11. สำหรับแต่ละเป้าหมาย:  encode_target_pose(...) → write_slot(...)
12. dump data/logs/position_mem.json  (สแนปช็อตผลลัพธ์เต็ม)
13. set_status(busy=0, complete=1); รอ PLC เคลียร์ trigger;
    ค้าง complete นาน complete_pulse_sec; set_status(ready=1, complete=0)
```

---

## `encode_target_pose()` — การแปลงเฟรมโดยละเอียด

นี่คือหัวใจทางคณิตศาสตร์ สำหรับหนึ่งเป้าหมาย มันทำ:

1. **อินพุต:** ค่าที่ transformer คืน `[X, -Y, -Z, roll, pitch, yaw, R_pcd]` (เฟรม point-cloud)
   ใช้ตำแหน่ง `[0:3]` และ **เมทริกซ์ 3×3 ที่ index 6** — ไม่ใช่ค่า Euler
2. **พลิกกลับเป็นเฟรม OpenCV camera:** `y_cam = -y_trans`, `z_cam = -z_trans`
   (เฟรม point-cloud มี Y ขึ้น / Z ถอยหลัง ส่วนเฟรม camera คือ Y ลง / Z ไปข้างหน้า)
3. **บวก `ee_offset` ของแท่ง** (ในเฟรมกล้อง) แล้วแปลงจุด Cam→Base:
   `cam_point_to_base(H_cam2base, …)`
4. **แปลงทิศทาง** Cam→Base: `R_base = H_cam2base[:3,:3] @ F @ R_pcd` โดยที่
   `F = diag(1,-1,-1)` พลิกเฟรม point-cloud กลับเป็นเฟรม raw camera
5. **เป็นมุม KUKA:** `Rotation.from_matrix(R_base).as_euler("ZYX", degrees=True)` →
   `A, B, C` (ตรงกับ convention intrinsic Z-Y-X ของ KUKA ที่ใช้ในการคาลิเบรต)
6. **เข้ารหัส** X,Y,Z (mm), A,B,C (deg), Conf เป็น int32 ×1000 → **14 words** ผ่าน
   `encode_pose` + `int32_to_words`

คืนค่า `(slot_words, (x,y,z) เมตร, (A,B,C) องศา)` — tuple สำหรับ logging และ JSON position-memory

> ตัวช่วยเรื่องเฟรมสองตัว `kuka_abc_to_matrix` / `build_cam2base` อยู่บนสุดของ `main.py`
> `kuka_abc_to_matrix` ใช้ convention Z-Y-X เดียวกับ `rotation_matrix_from_abc` ใน
> `tools/geometry.py` — ถ้าจะเปลี่ยนตัวใดตัวหนึ่ง ให้คงความสอดคล้องกันไว้

---

## ตัวช่วยสถานะ & error (บนสุดของ `main.py`)

- `set_status(plc, cfg, ready=…, busy=…, complete=…, error=…)` — อัปเดตเฉพาะบิตที่ส่งเข้าไป
  จำที่เหลือไว้ใน `_last_status` และส่งทั้งสี่บิตไป PLC ใน **หนึ่ง** แพ็กเก็ต `write_bits`
  ใช้ตัวนี้กับการเปลี่ยนสถานะทุกครั้ง
- `report_no_results(...)` — handshake มาตรฐานสำหรับ "trigger แล้วไม่ได้อะไร": amount=0,
  error no_targets, พัลส์ complete, รอ trigger เคลียร์, กลับสู่ ready
- `_wait_trigger_low(...)` — บล็อกจนกว่า PLC จะเคลียร์บิต trigger (ack ของมัน) timeout 10 วินาที

---

## ตอนปิดระบบ

เมื่อกด `q`/`ESC`/Ctrl-C บล็อก `finally` จะตั้ง `error=1` (เพื่อให้ PLC รู้ว่า PC ดับ),
ตัดการเชื่อมต่อ PLC (ซึ่งหยุด heartbeat), ปล่อยกล้อง, และปิดหน้าต่าง

---

## แบบจำลองในหัวอย่างรวบรัด

> **ลูปคือ: poll PLC → เมื่อ trigger ให้ถ่ายภาพ หาจุด แปลงแต่ละจุดเป็น pose 6-DOF ในเฟรม
> ฐานหุ่นยนต์ เขียนลงรีจิสเตอร์ PLC แล้ว handshake จบ** ทุกอย่างใน `core/`,
> `communication/` และตัวช่วย geometry คือเครื่องมือที่ลูปนี้เรียกใช้
