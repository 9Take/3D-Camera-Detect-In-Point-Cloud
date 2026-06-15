# ระเบียบวิธี (Solution Methodology)

> 🌐 ภาษา: [English](methodology.md) | **ไทย**

ระเบียบวิธีแบบครบวงจรของระบบ Vision สำหรับหุ่นยนต์นำทาง 3 มิติ (3D Guidance Robot): ตั้งแต่ trigger ของ PLC ไปจนถึง
pose 6-DoF เต็มที่เขียนกลับเข้ารีจิสเตอร์ PLC ระบบนี้ผสาน detector แบบจับคู่ template 2 มิติ
เข้ากับ depth จากกล้อง Intel RealSense, ยกแต่ละจุดที่เจอขึ้นเป็น pose 3 มิติ, แล้วแปลงจาก
**เฟรมกล้องไปเป็นเฟรมฐานหุ่นยนต์** (ผ่านการคาลิเบรต hand-eye) ก่อนส่งให้ PLC/หุ่นยนต์

---

## 1. ภาพรวมระบบ

```
   PLC  ──trigger / program no. / pose หุ่นยนต์สด──►  PC (Vision)
    ▲                                                   │  ──X,Y,Z,A,B,C,Conf (เฟรม base)──►  PLC
    │                                                   ▼
    └────────── status / heartbeat ──────────── RealSense D4xx (RGB + Depth)
```

สามชั้นที่ทำงานร่วมกัน:

| ชั้น | โมดูล | หน้าที่ |
|------|-------|---------|
| Acquisition (รับข้อมูล) | [communication/realsense.py](../communication/realsense.py) | เฟรม RGB-D ที่ align แล้ว, การกรอง depth |
| Perception (รับรู้) | [core/detector.py](../core/detector.py), [core/transformer.py](../core/transformer.py) | จับคู่ฟีเจอร์ 2D → pose 3D |
| Integration (เชื่อมรวม) | [communication/plc_comm.py](../communication/plc_comm.py), [main.py](../main.py) | การแปลง Cam→Base, MC-protocol I/O, handshake, สถานะ, error code |

การตั้งค่าตอนรันอยู่ใน [config.yaml](../config.yaml) (กล้อง, โปรแกรม, ที่อยู่อุปกรณ์ PLC,
การคาลิเบรต hand-eye, การสเกล)

> คู่มือส่งมอบงานที่จัดตามส่วนงานสำหรับวิศวกรคนใหม่อยู่ที่
> [handover/](handover/) ไฟล์นี้เป็นเอกสารอ้างอิงเชิงลึกที่ระบุเลขบรรทัด

---

## 2. คลัง Template ต่อโปรแกรม

"หมายเลขโปรแกรม" ที่มาจาก PLC เลือกว่าจะจับคู่ชุด template ชุดไหน แต่ละโปรแกรมคือโฟลเดอร์ที่มี
โฟลเดอร์ย่อย `Point*` หนึ่งตัวหรือมากกว่า และแต่ละจุดมี sub-template ได้หลายตัว เพื่อให้ฟีเจอร์
ทางกายภาพเดียวกันถูกจดจำได้ภายใต้มุมมอง/แสงที่ต่างกัน:

```
data/templates/ProgramA/
   PointA/
      A1_template.png
      A1_meta.json     # offset_x, offset_y → จุด "คลิก" ที่แน่นอนภายใน template
      A2_template.png
      A2_meta.json
   PointB/
      ...
```

กฎการโหลด ([core/detector.py:14-52](../core/detector.py#L14-L52)):

- สแกนเฉพาะไดเรกทอรีที่ขึ้นต้นด้วย `Point`
- สำหรับทุก `*_template.png` จะคำนวณ SIFT keypoints + descriptors ล่วงหน้าครั้งเดียวตอนเริ่มระบบ
- `*_meta.json` (ใช้ก่อน) หรือ `*_offset.txt` กำหนด *target offset* — พิกเซลภายใน template ที่
  ตรงกับจุดคลิกทางกายภาพ ถ้าไม่มีไฟล์ meta จะใช้จุดกึ่งกลางของ template

การโหลด `ObjectDetector` หนึ่งตัวต่อโปรแกรมล่วงหน้า ([main.py:32-40](../main.py#L32-L40))
ทำให้ latency ตอน trigger ต่ำ การสลับโปรแกรมตอนรันก็แค่ค้นใน dictionary

---

## 3. การรับเฟรม (RealSense)

[communication/realsense.py](../communication/realsense.py) ตั้งค่ากล้องให้ depth ใช้งานได้
บนพื้นผิวโลหะที่สะท้อนแสง (specular):

1. **visual preset แบบ High-Density** (`rs.option.visual_preset = 4`) — เพิ่มความครอบคลุมบน
   พื้นผิว texture ต่ำ/สะท้อนแสง
2. **ฟิลเตอร์ depth**: spatial → temporal → hole-filling ใส่กับทุกเฟรมดิบใน
   [communication/realsense.py:53-57](../communication/realsense.py#L53-L57)
3. **Align เข้ากับ color stream** เพื่อให้พิกเซล `(u, v)` ในภาพ RGB map ตรงไปยังค่า depth ที่
   `(u, v)` เดียวกันในภาพ depth

ความละเอียด 640×480 @ 30 FPS ทั้งสองสตรีม (ดู [config.yaml:1-3](../config.yaml#L1-L3))

---

## 4. การตรวจจับ 2 มิติ (SIFT + FLANN + Homography)

สำหรับแต่ละเฟรม [core/detector.py:54-103](../core/detector.py#L54-L103) ทำ:

1. แปลงเฟรมเป็นขาวดำ คำนวณ SIFT keypoints/descriptors
2. สำหรับทุก template ของโปรแกรมที่ใช้งาน:
   - **จับคู่ FLANN k-NN** (k=2) ระหว่าง descriptor ของ template กับเฟรม
   - **Lowe's ratio test** ที่ 0.7 เพื่อเก็บ "good" matches
   - ต้องมี **> 12 good matches** ก่อนจะลองหา pose
3. **RANSAC homography** `M` จากพิกัด template → เฟรม
4. **ค่าความมั่นใจ** มาจากจำนวน RANSAC inlier ตัดเพดานที่ 100%:
   `confidence = min(100, inliers / 30 * 100)`
5. ฉายจุด offset ของ template ผ่าน `M` เพื่อได้ **พิกเซลเป้าหมาย** `(u, v)` ในเฟรมสด และฉาย
   มุมทั้งสี่ของ template เพื่อวาดกรอบ polygon

### 4.1 การกรอง Best-per-Point

หนึ่ง `Point` อาจมี sub-template หลายตัวที่จับคู่ได้ทั้งหมด เก็บเฉพาะการตรวจจับที่ confidence
สูงสุดต่อจุด ([main.py:43-53](../main.py#L43-L53)) เพื่อไม่ให้เขียนซ้ำลงตาราง slot ของ PLC
สำหรับฟีเจอร์ทางกายภาพเดียวกัน

---

## 5. การยก 2D → 3D + pose 6-DoF

[core/transformer.py:21-121](../core/transformer.py#L21-L121) เปลี่ยนแต่ละพิกเซลที่ตรวจจับ
เป็น pose 6-DoF:

1. **อ่าน depth** ที่พิกเซลที่ตรวจจับ
2. **กู้ค่า depth** (แก้จุดบอดของกล้อง,
   [core/transformer.py:56-74](../core/transformer.py#L56-L74)): ถ้า depth เป็น 0 ให้ขยาย
   รัศมีค้นหาจาก 2 → 7 px แล้วเฉลี่ยเพื่อนบ้านที่ไม่เป็นศูนย์ ถ้าไม่พบ depth ที่ใช้ได้ในหน้าต่าง
   15×15 จะข้ามเป้าหมายนั้น (และรายงานขึ้นไปเป็น ERR_NO_TARGETS)
3. **Back-project** พิกเซลด้วย intrinsics ของ RealSense:
   ```
   Z = depth_raw * depth_scale
   X = (u - cx) * Z / fx
   Y = (v - cy) * Z / fy
   ```
4. **สร้าง point cloud** จากภาพ RGB-D ที่ align แล้ว
   ([core/transformer.py:34-42](../core/transformer.py#L34-L42)) และพลิก Y/Z เพื่อให้เฟรมโลก
   เป็น right-handed และ "ขึ้น = +Y, ไปข้างหน้า = -Z"
5. **ประมาณ normal ต่อจุด** ด้วย `KDTreeSearchParamHybrid(0.01, 30)`
6. **กำหนดทิศเฟรมเฉพาะที่**: ใช้ normal ของพื้นผิวเป็นแกน Z เฉพาะที่ เลือกค่าตั้งต้นที่เสถียร
   สำหรับ X (`[1,0,0]` หรือ `[0,1,0]` เพื่อเลี่ยง degeneracy) แล้ว `Y = Z × X`, `X = Y × Z`
7. **แปลง** เมทริกซ์การหมุนเป็น Euler `(roll, pitch, yaw)` ด้วย
   `rotation_matrix_to_euler_angles` ใน [core/transformer.py:6-13](../core/transformer.py#L6-L13)
8. คืนค่า `{target_name: [X, -Y, -Z, roll, pitch, yaw, rotation_matrix]}`
   ([core/transformer.py:99](../core/transformer.py#L99)) **index 6 บรรจุเมทริกซ์ทิศทาง 3×3
   เต็ม** (ในเฟรม point-cloud) `main.py` ใช้เมทริกซ์นั้น — ไม่ใช่มุม Euler — เพื่อแปลงทิศทางไป
   เฟรมฐานหุ่นยนต์ (§6) ส่วนค่า Euler เก็บไว้สำหรับ logging/template

พิกัดเหล่านี้อยู่ในเฟรม **point-cloud** (X ขวา, Y ขึ้น, Z ถอยหลัง) `main.py` พลิก Y/Z กลับเป็น
เฟรม OpenCV camera ก่อนทำการแปลง Cam→Base

point cloud, ลูกบอลเป้าหมาย, และแกนพิกัด ถูกแคชไว้ที่ transformer (`_last_geometries`) เพื่อ
ให้ผู้ใช้เปิด Open3D viewer ด้วยปุ่ม `p` ได้โดยไม่ต้องรันการตรวจจับใหม่

---

## 6. การแปลงเฟรมกล้อง → ฐานหุ่นยนต์ (Hand-Eye)

detector/transformer ผลิต pose ในเฟรม **กล้อง** แต่หุ่นยนต์ต้องการมันในเฟรม **base** ของตัวเอง
การแปลงนี้คือครึ่งหลังของ `main.py` และเป็นสิ่งที่ทำให้เอาต์พุตใช้กับ KUKA ได้โดยตรง

### 6.1 ห่วงโซ่ 4×4

`build_cam2base` ([main.py:36-48](../main.py#L36-L48)) ประกอบการแปลงสองตัวเข้าเป็น
`H_cam2base` ก้อนเดียว:

```
H_cam2base = H_gripper2base @ H_cam2gripper
```

- **`H_cam2gripper`** — ผล **คาลิเบรต hand-eye** ที่คงที่
  (`robot.hand_eye_rotation` / `hand_eye_translation` ใน config) ผลิตโดย
  [calibration/aruco_calibate.py](../calibration/aruco_calibate.py); ดู
  [handover/calibration.th.md](handover/calibration.th.md) สำหรับวิธีรัน/ตรวจสอบ
- **`H_gripper2base`** — **pose ถ่ายภาพ** ของหุ่นยนต์ (ที่แขนจอดตอนถ่ายภาพ) สร้างจาก
  `(x,y,z,a,b,c)` ด้วย `kuka_abc_to_matrix`
  ([main.py:27-33](../main.py#L27-L33)) ซึ่งใช้ convention **intrinsic Z-Y-X** ของ KUKA
  (`Rz(A)·Ry(B)·Rx(C)`)

### 6.2 pose ถ่ายภาพสดจาก PLC

pose ถ่ายภาพถูก **อ่านสดจาก PLC ทุก trigger**
(`read_robot_scan_pose`, [main.py:51-59](../main.py#L51-L59)) จาก `pose_device`
(`D2000`, int32 หกตัว ×1000, mm/deg) แล้วสร้าง `H_cam2base` ใหม่ ส่วน `robot.scan_pose`
แบบคงที่ใน config เป็นเพียง **fallback** ที่ใช้ตอนเริ่มหรือถ้าการอ่าน PLC fail / คืนเป็นศูนย์
ทั้งหมด — ตั้งค่าให้เป็น pose ที่จอดจริง เพื่อให้การอ่านที่ fail ไม่ส่งหุ่นยนต์ไปผิดที่

### 6.3 การเข้ารหัสต่อเป้าหมาย

`encode_target_pose` ([main.py:75-114](../main.py#L75-L114)) เปลี่ยนหนึ่งเป้าหมายที่ตรวจจับ
เป็น slot ผลลัพธ์ของ PLC:

1. พลิกพิกัดเฟรม point-cloud กลับเป็น **เฟรม OpenCV camera** (`y_cam = -y`, `z_cam = -z`)
2. บวก **`ee_offset`** ของแท่ง (เฟรมกล้อง, เมตร) แล้วแปลงจุด Cam→Base
3. แปลงทิศทาง Cam→Base: `R_base = H_cam2base[:3,:3] · F · R_pcd` โดยที่
   `F = diag(1,-1,-1)` ยกเลิกการพลิก point-cloud จากนั้น
   `Rotation.from_matrix(R_base).as_euler("ZYX", degrees=True)` → KUKA `A,B,C`
4. เข้ารหัส `X,Y,Z` (mm), `A,B,C` (deg) และ Confidence เป็น **int32 ×1000**
   (low-word-first) ผ่าน [tools/plc_decode.py](../tools/plc_decode.py) → 14 words

`ee_offset` เลื่อนเฉพาะตำแหน่ง (เพื่อให้จุดที่รายงานไปตกที่ปลายแท่ง) ทิศทางการหมุนไม่เปลี่ยน

---

## 7. การเชื่อมรวมกับ PLC

### 7.1 โปรโตคอล

[communication/plc_comm.py](../communication/plc_comm.py) พูดภาษา **MELSEC
MC-protocol (Type3E binary)** ผ่าน TCP ที่ที่อยู่ใน
[config.yaml:36-37](../config.yaml#L36-L37) การเข้าถึงทั้งหมดถูกห่อด้วย thread lock พร้อม
reconnect ที่จำกัดอัตรา (cooldown 2 วินาที) เพื่อให้การหลุดชั่วคราวไม่ทำให้ลูปพัง

### 7.2 Handshake (หนึ่งรอบ trigger)

```
PLC                                  PC
 ─ write D1100 = program_no   ──►  poll (D1100)        sticky
 ─ write D2000 = pose หุ่นยนต์  ──►  (อ่านตอน trigger)
 ─ set  M1500 = 1 (trigger)   ──►  read  (M1500)
                                   set status: ready=0, busy=1
                                   อ่าน pose สด (D2000), สร้าง Cam→Base ใหม่
                                   จับเฟรม, ตรวจจับ, ยกเป็น 3D, แปลง
                                   write D1002 = amount
                                   write D1003.. = X,Y,Z,A,B,C,Conf ต่อ slot
                                   set status: busy=0, complete=1
                              ◄──   รอ M1500 = 0 (PLC ack)
 ─ clear M1500 = 0           ──►  ค้าง complete=1 เป็นเวลา COMPLETE_PULSE_SEC
                                   set status: ready=1, complete=0
```

พัลส์ `complete=1` ถูกค้างไว้ `complete_pulse_sec` (1 วินาที) เพื่อให้ HMI ที่ scan ช้ายัง
latch ได้ทัน

### 7.3 การแพ็กสถานะ (ลด latency)

บิตสถานะทั้งสี่ (`ready, error, busy, complete`) map ไปยังอุปกรณ์ M ติดกันสี่ตัว
(`M1000..M1003`) และส่งใน **หนึ่ง** แพ็กเก็ต `write_bits` แทนการเขียนแยกสี่ครั้ง (`set_status`,
[main.py:155-168](../main.py#L155-L168)) แคชในกระบวนการขนาดเล็ก (`_last_status`) ทำให้
อัปเดตบางส่วนได้โดยไม่ต้องอ่านเพิ่ม

### 7.4 การสเกลหน่วยจริง → INT32

ค่า pose แต่ละค่าถูกส่งเป็น **จำนวนเต็ม 32 บิตที่แยกเป็นสองคำ 16 บิต** (low-word-first)
สเกล ×1000 — เป็นการเข้ารหัสเดียวกับที่ PLC ใช้ส่ง pose หุ่นยนต์มาให้เรา ดังนั้น
[tools/plc_decode.py](../tools/plc_decode.py) จึง round-trip ได้ทั้งสองทิศทาง:

| ค่า | การสเกล | เข้ารหัสเป็น |
|-----|---------|--------------|
| X / Y / Z | × 1000 | เมตร → mm (×1000 → int32 ความละเอียดระดับ µm) |
| A / B / C | × 1000 | องศา → มิลลิองศา int32 |
| Confidence | × 100 | เปอร์เซ็นต์ × 100, int32 |

> หมายเหตุ: ตอนนี้ `config.yaml` ตั้ง `words_per_slot: 14` ให้ตรงกับดีไซน์ปัจจุบันแล้ว
> แต่ค่านี้เป็นเอกสารประกอบเท่านั้น — `main.py` hard-code **14 words/slot**
> ([main.py:358](../main.py#L358)) และสเกลผ่าน `plc_decode` ส่วน key `position_multiplier: 10000`
> ที่ตกค้างนั้นไม่ถูกใช้งาน (สเกลจริงคือ ×1000 ผ่าน `POSE_SCALE` ใน `plc_decode`)
> และลบทิ้งได้อย่างปลอดภัย

โครงสร้าง slot จาก `D1003` (`slot_base_device`), **14 words (7 int32) ต่อ slot**:

```
slot k  →  D1003 + k*14
   +0  X    +2  Y    +4  Z
   +6  A    +8  B    +10 C
   +12 Conf
...                          สูงสุด max_points = 5
```

### 7.5 Error Codes (`error_code_device`, `D1001`)

| Code | ความหมาย |
|------|----------|
| 0    | OK |
| 1    | หมายเลขโปรแกรมไม่ถูกต้อง |
| 2    | ไม่พบเป้าหมาย |
| 3    | อ่านกล้อง fail |
| 99   | error ภายใน |

### 7.6 Heartbeat

thread เบื้องหลังเพิ่ม counter และเขียนไปที่ `D1000` ทุกวินาที
(`start_heartbeat`,
[communication/plc_comm.py:149-164](../communication/plc_comm.py#L149-L164)) เพื่อให้ PLC
ตรวจจับ PC ที่ตายไปแล้ว

---

## 8. ลูปหลัก (orchestration)

ลูป `main()` ([main.py:336-572](../main.py#L336-L572)) คือตัวขับวงจร ต่อหนึ่งรอบ:

1. ดึงเฟรม RGB-D ใหม่จากกล้อง
2. **poll PLC แบบจำกัดอัตรา** (10 Hz โดยปริยาย เป็นอิสระจาก FPS ของกล้อง) อ่านหมายเลขโปรแกรม
   และบิต trigger การ poll ที่ FPS ของกล้องจะทำให้ลิงก์ PLC ล้น
3. เรนเดอร์พรีวิวสดพร้อมการตรวจจับของโปรแกรมที่ใช้งาน (กรอบ + จุดเป้าหมาย + ป้ายต่อจุด *best*)
4. เมื่อ trigger:
   - ล็อก `current_program_no`, ตั้ง `busy=1`
   - **อ่าน pose ถ่ายภาพสดจาก PLC** และสร้าง `H_cam2base` ใหม่ (§6.2); fallback ไปใช้ pose
     จาก config ถ้าใช้ไม่ได้
   - จับเฟรมใหม่สำหรับการสแกนจริง (เฟรมนิ่งน่าเชื่อถือกว่าเฟรมพรีวิวที่ใช้แสดงผล)
   - รัน detector → กรอง best-per-point → ยก 3D
   - แสดง grid "trigger result" หนึ่ง tile ต่อ sub-template โดย tile BEST ตีกรอบเขียว
     (`_build_trigger_result_grid`, [main.py:271-330](../main.py#L271-L330))
   - เขียน `amount` แล้วเขียนต่อ slot `(X, Y, Z, A, B, C, Conf)` หลังการแปลง Cam→Base (§6.3)
   - บันทึกสแนปช็อต JSON ไปที่ `data/logs/position_mem.json` เพื่อตรวจสอบย้อนหลัง
   - พัลส์ `complete=1`, รอ PLC เคลียร์ trigger, แล้วกลับสู่ `ready=1`

### 8.1 โหมด Debug (`--debug`)

เพิ่มการควบคุมด้วยคีย์บอร์ดเพื่อทำงานโดยไม่ต้องมี PLC:

- `1`–`9`: เลือกโปรแกรมด้วยมือ
- `t`: trigger ด้วยมือ
- `b`: เข้าโหมดย่อย PLC-test — `1`–`9` เขียนไปที่ `program_no_test_device` และ `t` พัลส์
  `trigger_test_device` เพื่อให้วิศวกร PLC ตรวจสอบฝั่งของตนในการ handshake
- `p`: เปิด 3D viewer ที่แคชไว้ของการสแกนล่าสุด

---

## 9. โหมดความล้มเหลวและการรับมือ

| ความล้มเหลว | จัดการที่ไหน |
|-------------|--------------|
| PLC TCP หลุด | `_try_reconnect` พร้อม cooldown 2 วินาที ([communication/plc_comm.py:57-73](../communication/plc_comm.py#L57-L73)) |
| บิต trigger ค้าง (cache) | trigger ถูกบริโภคในเครื่อง ลูปรอให้ PLC เคลียร์ก่อนรอบถัดไป (`_wait_trigger_low`, [main.py:183-192](../main.py#L183-L192)) |
| Depth = 0 ที่พิกเซลเป้าหมาย (ทองแดงสะท้อนแสง) | ค่าเฉลี่ยเพื่อนบ้านขยายรัศมี ([core/transformer.py:56-74](../core/transformer.py#L56-L74)) |
| pose หุ่นยนต์เป็นศูนย์ทั้งหมด / อ่าน PLC fail | fallback ไปใช้ `scan_pose` จาก config สำหรับ Cam→Base ([main.py:456-465](../main.py#L456-L465)) |
| sub-template จับคู่ซ้ำ | กรอง best-per-point (`best_per_point`, [main.py:137-147](../main.py#L137-L147)) |
| โปรแกรมไม่ถูกโหลดแต่ trigger ยิง | `ERR_INVALID_PROGRAM`, สถานะ `error=1` ([main.py:443-448](../main.py#L443-L448)) |
| HMI scan ช้าพลาดพัลส์ `complete` | ค้าง `complete_pulse_sec` ([main.py:561](../main.py#L561)) |
| อัตราแพ็กเก็ต PLC มากเกินไป | จำกัดด้วย `poll_interval_sec` + เขียนสถานะแบบแพ็กรวม ([main.py:382-383](../main.py#L382-L383), [155-168](../main.py#L155-L168)) |

---

## 10. เส้นทางข้อมูลแบบครบวงจร (หนึ่ง trigger)

```
PLC trigger  ─►  อ่าน pose ถ่ายภาพหุ่นยนต์สด (D2000) → สร้าง H_cam2base ใหม่
              ─►  เฟรม RGB+Depth  ─►  จับคู่ SIFT/FLANN กับ template ของ ProgramN
              ─►  RANSAC homography → พิกเซลเป้าหมาย (u,v) + confidence
              ─►  กรอง best-per-point
              ─►  อ่าน depth (พร้อมกู้ค่า 0) → back-projection → (X,Y,Z)
              ─►  point cloud + ประมาณ normal → pose 6-DoF (เฟรมกล้อง)
              ─►  แปลง Cam→Base (+ ee_offset) → X,Y,Z,A,B,C (เฟรม base)
              ─►  เข้ารหัส int32 ×1000 → เขียน amount + slot 14 คำ
              ─►  ตั้ง complete=1, รอ PLC ack, กลับสู่ ready
```
