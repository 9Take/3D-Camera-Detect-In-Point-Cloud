# 6. Configuration — `config.yaml`

> 🌐 ภาษา: [English](configuration.md) | **ไทย**

การตั้งค่าตอนรันทั้งหมดอยู่ใน [`config.yaml`](../../config.yaml) ไม่มีค่าคงที่ที่ฝังใน `main.py`
ที่ปกติคุณจะต้องไปเปลี่ยน (มีค่าจูน detector สองสามตัวอยู่ใน `core/detector.py` — ดู
[core.th.md](core.th.md)) หน้านี้อธิบายทุกบล็อก

---

## `camera:`
```yaml
camera:
  resolution_width: 640
  resolution_height: 480
```
ขนาดสตรีมของ RealSense ทั้ง color และ depth 640×480 @ 30 FPS คือค่าที่ผ่านการทดสอบ

---

## `charuco:` — เรขาคณิตของ board คาลิเบรต
```yaml
charuco:
  squares_x: 7          # ช่องตามแนวนอน
  squares_y: 5          # ช่องตามแนวตั้ง
  square_length: 0.0494 # เมตร, ด้านของช่องหมากรุก
  marker_length: 0.025  # เมตร, ด้านของ ArUco marker
  dictionary: DICT_6X6_250
```
**ต้องตรงกับ board จริงที่พิมพ์ออกมาเป๊ะ ๆ** ใช้เฉพาะการคาลิเบรต (หมายเหตุ: board 7×5 มี
6×4 = 24 *มุม* — อย่าสับสนระหว่าง id มุมกับ id marker)

---

## `calibration:` — การตั้งค่าการเก็บข้อมูล hand-eye
```yaml
calibration:
  total_poses: 16          # จำนวน pose ที่เก็บก่อนแก้สมการ
  min_charuco_corners: 8   # มุมขั้นต่ำสำหรับ pose ของ board ที่เสถียร
  max_reproj_px: 2.0       # ปฏิเสธ pose board ที่แย่กว่านี้ (ฆ่าการพลิก 180°)
  settle_sec: 0.5          # รอหลัง trigger ก่อนบันทึก (ให้หุ่นยนต์นิ่ง)
  pose_stable_tol: 0.5     # ความเปลี่ยนแปลงสูงสุด (mm/deg) ระหว่างสองครั้งที่อ่าน pose ติดกัน
  debug_raw_pose: true     # พิมพ์คำดิบ + การถอดรหัสทั้งสองแบบทุกครั้งที่จับ
  debug_handshake: true    # พิมพ์บิต trigger/ack + การเปลี่ยน state
```
ดู [calibration.th.md](calibration.th.md)

---

## `programs:` — ตัวเลือกคลัง template
```yaml
programs:
  1: { name: ProgramA, template_dir: data/templates/ProgramA }
  2: { name: ProgramB, template_dir: data/templates/ProgramB }
  ...
```
ตัวเลขคือสิ่งที่ PLC ส่งมาใน `program_no_device` `main.py` โหลด `ObjectDetector` หนึ่งตัวต่อ
หนึ่งโปรแกรมตอนเริ่มระบบ ทำให้สลับตอนรันได้ทันที แต่ละ `template_dir` มีโฟลเดอร์ย่อย `Point*/`
ของ template ที่สอนไว้

---

## `plc:` — ข้อตกลงกับ PLC
นี่คือบล็อกใหญ่ ดู [communication.th.md](communication.th.md) §3.3 สำหรับตารางรีจิสเตอร์เต็มและ
ความหมายของแต่ละอุปกรณ์ ส่วนที่สำคัญ:

```yaml
plc:
  ip: 192.168.1.165
  port: 5010

  trigger_device: M1500       # PLC→PC: เริ่มสแกน
  program_no_device: D1100    # PLC→PC: โปรแกรมไหน
  use_live_scan_pose: false   # true = อ่าน pose หุ่นยนต์สดจาก PLC ทุก trigger;
                              # false = ใช้ค่า robot.scan_pose ที่ตั้งมือด้านล่างเสมอ
  pose_device: D2000          # PLC→PC: pose หุ่นยนต์สด (6× int32)
  pose_word_count: 12
  pose_word_swap: false       # ตั้ง true ถ้า KUKA REAL ส่งกลับมาเป็น high-word-first

  calib_trigger_device: M2000 # handshake คาลิเบรตเท่านั้น
  calib_ack_device: M2001

  heartbeat_device: D1000     # สถานะ PC→PLC
  error_code_device: D1001
  status_ready_device: M1000  # ready/error/busy/complete ต้อง "ติดกัน"
  status_error_device: M1001
  status_busy_device: M1002
  status_complete_device: M1003

  amount_device: D1002        # ผลลัพธ์ PC→PLC
  slot_base_device: D1003
  words_per_slot: 14          # 7 int32: X Y Z A B C Conf (ตรงกับ main.py)
  max_points: 5

  heartbeat_interval_sec: 1.0
  poll_interval_sec: 0.1      # ความถี่ในการ poll trigger/program ของ PLC
  complete_pulse_sec: 1.0     # ค้าง complete=1 นานเท่านี้หลัง ack
  position_multiplier: 10000
  confidence_multiplier: 100
  error_codes: { ok: 0, invalid_program: 1, no_targets: 2, camera: 3, internal: 99 }
```

> หมายเหตุ: ตอนนี้ `words_per_slot` เป็น `14` ให้ตรงกับ `main.py` แล้ว แต่ค่านี้เป็นเอกสาร
> ประกอบเท่านั้น — `main.py` hard-code **14 words/slot** และส่ง 6-DOF เต็ม (X Y Z A B C Conf)
> เป็น int32 ส่วน `position_multiplier` (10000) เป็นของตกค้างจากเส้นทาง int16 เก่าและไม่ถูกใช้งาน
> — โค้ดปัจจุบันสเกล pose ด้วย ×1000 ผ่าน `tools/plc_decode.py` เชื่อโค้ด: **14 words/slot, int32 ×1000**

---

## `robot:` — ผลคาลิเบรต + เรขาคณิต
```yaml
robot:
  hand_eye_rotation:    [...3x3...]   # R_cam2gripper จากการคาลิเบรต (ไม่มีหน่วย)
  hand_eye_translation: [...3...]     # t_cam2gripper, เมตร (คาลิเบรตพิมพ์เป็น mm → ÷1000)

  scan_pose:                          # pose ถ่ายภาพ (เฟรม WORLD ของ SmartPAD)
    x: 0.530  # m                     # ใช้ค่านี้ตรง ๆ เมื่อ use_live_scan_pose = false
    y: 0.015  # m                     # เมื่อเปิดโหมดสด ค่านี้เป็นค่าเริ่มต้นตอนสตาร์ต และ
    z: 0.090  # m                     # เป็น fallback ถ้าอ่าน pose จาก PLC fail / เป็นศูนย์
    a: -90.0  # deg                   # ตั้งค่าให้เป็น pose ถ่ายภาพที่จอดจริง
    b: 0.0
    c: -180.0

  ee_offset:                          # offset ปลายแท่ง end-effector จากกล้อง วัดในเฟรมกล้อง
    x: 0.0   # m                      # (X ขวา, Y ลง, Z ไปข้างหน้า) บวกเข้ากับแต่ละจุดที่ตรวจจับ
    y: 0.0   # m                      # เพื่อให้ X/Y/Z ที่รายงานไปตกที่ปลายแท่ง ทิศทางการหมุน
    z: -0.295 # m                     # ไม่ถูกเปลี่ยนโดยตัวนี้
```
- **`hand_eye_*`** — ได้จากการรันคาลิเบรต เป็นการส่งต่อด้วยมือเพียงจุดเดียว ดู
  [calibration.th.md](calibration.th.md)
- **`use_live_scan_pose`** — เลือกว่าจะเอา pose ถ่ายภาพจากไหน `false` (ค่าเริ่มต้น): ใช้
  `scan_pose` ที่ตั้งมือด้านล่างเสมอ `true`: `main.py` อ่าน pose *สด* จาก PLC
  (`pose_device`) ทุก trigger แล้วสร้าง Cam→BASE ใหม่จากค่านั้น
- **`scan_pose`** — pose ถ่ายภาพที่ตั้งมือ ใช้ตรง ๆ เมื่อ `use_live_scan_pose` เป็น `false`;
  ในโหมดสดเป็นค่าเริ่มต้นตอนสตาร์ตและเป็น fallback ถ้าอ่านจาก PLC fail / เป็นศูนย์ทั้งหมด
- **`ee_offset`** — เลื่อนจุดที่รายงานจากศูนย์กลางออปติคัลของกล้องไปยังปลายเครื่องมือจริง
  ปรับถ้าหุ่นยนต์ไปตกแบบ offset ตามแกนใดแกนหนึ่งเสมอ ๆ

---

## `paths:`
```yaml
paths:
  debug_dir: data/templates/debug   # ที่ create_template --debug บันทึกไฟล์ .ply
  save_dir: data/logs
  position_mem: data/logs/position_mem.json   # ผลเต็มของสแกนล่าสุด (เขียนทุกรอบ)
```
