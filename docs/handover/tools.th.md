# 5. Tools — สคริปต์ช่วยเหลือ

> 🌐 ภาษา: [English](tools.md) | **ไทย**

โฟลเดอร์ `tools/` เก็บตัวช่วยที่ทำงานแบบ standalone สามตัว (`plc_decode`, `geometry`,
`board_detect`) เป็น **ไลบรารีล้วน** ที่ถูก import โดยตอนรันจริง/การคาลิเบรต ส่วนอีกสองตัว
เป็น **สคริปต์ที่คุณรันเอง**

---

## 5.1 `create_template.py` — สอน template ใหม่ (รันเอง)

นี่คือวิธีเพิ่มจุดใหม่ให้ detector ค้นหา เป็นเครื่องมือแบบ interactive ที่คลิกเลือกจุดบนภาพสดจากกล้อง

```bash
python tools/create_template.py -p ProgramA --point PointA -v 1
# หรือรัน `python tools/create_template.py` เฉย ๆ แล้วตอบคำถามที่ถาม
```

อาร์กิวเมนต์:
- `-p/--program` — หมายเลขโปรแกรม (`1`) หรือชื่อ (`ProgramA`) อ้างอิงกับ `config.yaml`
- `--point` — ชื่อจุด (`PointA`; ถ้าใส่ `A` เฉย ๆ จะกลายเป็น `PointA`)
- `-v/--variant` — แท็ก variant (`1`, `front`, `side`…) ชื่อที่บันทึกคือ `PointA.1`
- `--debug` — บันทึกไฟล์ `.ply` และแสดงการ align 3D ด้วย

ขั้นตอนบนหน้าจอ:
1. **Live view** → กด **SPACEBAR** เพื่อแช่ภาพ
2. **Annotate**: **คลิกซ้าย** เพื่อวาดกรอบ polygon รอบฟีเจอร์ **คลิกขวา** เพื่อระบุจุดเป้าหมาย
   ที่แน่นอน (จุดคลิก) `s` = บันทึก, `r` = ถ่ายใหม่, `c` = ล้าง
3. **Tracking**: มันจะจับคู่ template ใหม่ของคุณแบบสด เพื่อให้คุณยืนยันว่าล็อกได้ กด **`q`**
   เพื่อคำนวณข้อมูล 3D และบันทึก meta JSON หรือ **ESC** เพื่อออก

มันเขียนลงใน `data/templates/<Program>/<Point>/`:
- `PointA.1_template.png` — patch ขาวดำที่ครอป
- `PointA.1_meta.json` — `offset_x/offset_y` (จุดคลิกภายใน patch **นี่คือสิ่งที่ detector
  ตอนรันจริงอ่าน**) บวกกับตำแหน่ง/ทิศทาง 3D ที่จับได้ไว้เป็นบันทึก

> detector ตอนรันจริงต้องการแค่ `_template.png` + `offset_x/offset_y` ใน `_meta.json`
> ส่วน field 3D ใน meta เป็นข้อมูลประกอบเท่านั้น

---

## 5.2 `cal_ressult_calib.py` — ตัว debug คาลิเบรตแบบ offline (รันเอง)

เมื่อการคาลิเบรต hand-eye ให้ residual แย่ ให้รันตัวนี้ **โดยไม่ต้องมีหุ่นยนต์** เพื่อหาสาเหตุ
มันอ่าน `calibration/capture_log.csv` + `calibration/hand_eye_result.npz` แล้ว:

1. **Pair diagnostic** — สำหรับแต่ละคู่ pose ที่ติดกัน เปรียบเทียบมุมการหมุนสัมพัทธ์ของ gripper
   กับของกล้อง ทั้งสองต้องตรงกัน (`AX=XB`) ความต่าง > 8° จะ flag pose ที่ **พลิกหรือ desync**
   และบอกตัวที่แย่ที่สุดให้ทิ้ง/ถ่ายใหม่
2. **Brute-force** — ลองทั้ง 5 วิธี hand-eye × 4 ลำดับเมทริกซ์ × 8 ชุดเครื่องหมาย (32 convention)
   แล้วรายงานชุดที่ให้ residual ต่ำสุด

```bash
python tools/cal_ressult_calib.py
```

ใช้ตอบคำถาม: "ข้อมูลแย่ หรือเป็น solver/convention?" (pair diagnostic มองไม่เห็น error เรื่อง
ทิศทาง/transpose และความเปราะของ solver — ดังนั้น diagnostic เขียวแต่แก้สมการ fail ชี้ไปที่
**solver หรือ translation** ไม่ใช่การจับคู่การหมุน)

> หมายเหตุ: คอมเมนต์หัวไฟล์ของสคริปต์นี้อ้างถึงไฟล์ `cal.py` แต่ตรรกะ brute-force สด ๆ
> ตอนนี้อยู่ที่นี่ ใน `tools/cal_ressult_calib.py`

---

## 5.3 `plc_decode.py` — int32 ↔ คำของ PLC (ไลบรารี, ตรรกะล้วน)

แหล่งความจริงเดียวว่าจะแพ็ก pose/ผลลัพธ์ของหุ่นยนต์เป็นคำของ PLC อย่างไร ไม่ต้องเชื่อมต่อ PLC
จึงเทสต์ได้ด้วย list ของคำที่สร้างเอง

| ฟังก์ชัน | ทำอะไร |
|----------|--------|
| `decode_pose(words, swap=False)` | คำที่ติดกัน → tuple ของ float แต่ละค่า = 2 คำ (low, high), int32, ÷1000, ขยายเครื่องหมาย |
| `encode_pose(values, swap=False)` | float mm/deg → list คำ int16 (ผกผันของ decode) |
| `int32_to_words(value, swap=False)` | int32 หนึ่งค่า → `[low, high]` เป็น int16 มีเครื่องหมาย |

`POSE_SCALE = 1000.0` — PLC ส่ง/รับค่า pose สเกล ×1000 (mm→µm, deg→mdeg)

**`swap`** = ลำดับคำ `swap=True` หมายถึง high-word-first ตรงกับ `config.yaml → plc.pose_word_swap`
ถ้า pose ที่ถอดออกมาดูเป็นค่าขยะ/ใหญ่ผิดปกติ flag นี้คือผู้ต้องสงสัยอันดับแรก

---

## 5.4 `geometry.py` — คณิตศาสตร์มุม KUKA (ไลบรารี, numpy ล้วน)

| ฟังก์ชัน | ทำอะไร |
|----------|--------|
| `rotation_matrix_from_abc(A, B, C)` | Euler KUKA (deg) → เมทริกซ์การหมุน ABC ของ KUKA เป็น **intrinsic Z-Y-X**: `Rz(A)@Ry(B)@Rx(C)` ซึ่ง `"ZYX"` ของ scipy ตรงกัน |
| `marker_positions_in_base(...)` | map จุดกำเนิด marker เข้าเฟรม base ของแต่ละ pose (สำหรับเช็ก residual) |
| `residual_stats(pts_base)` | `(mean_point, rms_mm, max_mm)` การกระจายของจุดเหล่านั้น = error การแก้สมการ |

ไม่มี OpenCV ไม่มีฮาร์ดแวร์ — import และเทสต์ได้ทุกที่อย่างปลอดภัย

---

## 5.5 `board_detect.py` — pose ของ ChArUco board (ไลบรารี, ต้องใช้ OpenCV)

ใช้โดยการคาลิเบรตเพื่อหา board และกู้ pose ของมัน

| ฟังก์ชัน | ทำอะไร |
|----------|--------|
| `build_charuco(sx, sy, sq_mm, mk_mm, dict)` | สร้าง `(board, detector)` ที่ใช้ทุกที่ |
| `detect_board_pose(frame, ...)` | ตรวจจับ board, แก้ pose, คืน `(success, R_target2cam, t_target2cam, debug_frame, n_corners, reproj_px)` |

รายละเอียดสำคัญที่ฝังอยู่:
- **ข้าม `matchImagePoints()` ที่เสีย** (OpenCV 4.7.0) — สร้างคู่ obj↔img จาก
  `getChessboardCorners()[ids]` ดู [calibration.th.md](calibration.th.md)
- ใช้ **solver IPPE แบบระนาบ** ซึ่งคืน *ทั้งสอง* คำตอบกระจกของ board แบน เก็บตัวที่ reprojection
  error ต่ำกว่า → เอาชนะการพลิก ~180°
- ปฏิเสธคำตอบที่ reproj > `max_reproj_px` หรือ board Z อยู่นอก 50–3000 mm
