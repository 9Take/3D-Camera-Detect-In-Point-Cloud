# 2. Core — สมองของระบบ Vision

> 🌐 ภาษา: [English](core.md) | **ไทย**

แพ็กเกจ `core/` คือที่ที่พิกเซลกลายเป็น pose 3 มิติ มีสองไฟล์:

- [`core/detector.py`](../../core/detector.py) — **2D**: หาฟีเจอร์ที่สอนไว้ในภาพสี
- [`core/transformer.py`](../../core/transformer.py) — **2D → 3D**: เปลี่ยนพิกเซลที่เจอเป็น pose 6-DOF

---

## 2.1 `detector.py` — `ObjectDetector`

### หน้าที่
รับภาพสีเข้ามา แล้วหา template ที่สอนไว้ทุกตัวในภาพ คืนค่าสำหรับแต่ละจุดที่เจอ:
**พิกเซลเป้าหมาย** `(u, v)`, **ชื่อ template**, **ค่าความมั่นใจ** %, และ
**กรอบ polygon** (มุมจาก homography) ไว้สำหรับวาด

### หลักการทำงาน
1. **ตอนเริ่มระบบ** — `_load_templates(template_dir)` สำรวจโฟลเดอร์โปรแกรม สำหรับทุก
   `*_template.png` ที่อยู่ในโฟลเดอร์ย่อย `Point*/` มันจะ:
   - คำนวณ SIFT keypoints + descriptors ล่วงหน้า **ครั้งเดียว** (เพื่อให้การจับคู่ต่อเฟรมเร็ว)
   - อ่าน **target offset** (จุด "คลิก" ที่แน่นอนภายใน template) จากไฟล์ `*_meta.json`
     ที่คู่กัน (`offset_x`, `offset_y`) ถ้าไม่มีจะ fallback ไป `*_offset.txt` แล้วไปที่
     จุดกึ่งกลางของ template
   - เก็บ `{img, offset, kp, des, point}` โดยใช้ชื่อ target เป็น key ส่วน `point` คือ
     โฟลเดอร์แม่ `PointA`/`PointB` — ใช้ภายหลังในการเลือก "best per point"
2. **ต่อหนึ่งเฟรม** — `detect(color_frame, w, h)`:
   - ทำ SIFT บนเฟรมขาวดำ
   - สำหรับแต่ละ template: จับคู่ **FLANN k-NN (k=2)** + **Lowe ratio test (0.7)** เพื่อ
     เก็บ good matches ต้องได้ **> 12** good matches จึงจะไปต่อ
   - **RANSAC homography** จาก template → เฟรม
   - **ค่าความมั่นใจ** = `min(100, inliers / 30 * 100)`
   - ฉายจุด offset ของ template ผ่าน homography → ได้พิกเซลเป้าหมายในเฟรมสด ฉายมุมทั้ง 4
     → ได้กรอบ polygon
   - เก็บผลเฉพาะเมื่อพิกเซลเป้าหมายอยู่ในเฟรม

### ค่าที่คืนกลับ (ลำดับสำคัญ — ผู้เรียกแกะตามตำแหน่ง)
```python
detected_pixels, detected_names, detected_confidences, detected_homographies, display_frame
```

### `build_sub_window_grid(...)`
สร้าง "grid ภาพย่อ" ที่ครอปจากการตรวจจับทั้งหมด เพื่อใช้รีวิวด้วยสายตา (หนึ่ง tile ต่อหนึ่งจุด
พร้อมกรอบ + ค่าความมั่นใจ) ส่วน `main.py` มีเวอร์ชันที่ละเอียดกว่า (`_build_trigger_result_grid`)
ที่ไฮไลต์ best per point ด้วย ดังนั้นเมธอดนี้เป็นตัวรอง

### ปุ่มจูน (hard-code ไว้ใน `detect()`)
| อะไร | ค่า | ตำแหน่ง |
|------|-----|---------|
| Lowe ratio | `0.7` | บรรทัด ~71 |
| good matches ขั้นต่ำ | `> 12` | บรรทัด ~73 |
| RANSAC reproj threshold | `5.0` px | `findHomography` |
| การสเกล confidence | `inliers / 30` | บรรทัด ~81 |

ถ้าการตรวจจับเบาบางเกินไป ให้ผ่อน ratio (เช่น 0.75) หรือลด good matches ขั้นต่ำ ถ้าจับผิดบ่อย
ให้ทำให้เข้มขึ้น

---

## 2.2 `transformer.py` — `PointCloudTransformer`

### หน้าที่
รับรายการพิกเซลเป้าหมาย + ชื่อ แล้วอ่าน depth, back-project เป็น 3D, ประมาณทิศทางพื้นผิวเฉพาะที่
และคืน pose 6-DOF **ต่อหนึ่งเป้าหมาย** ในเฟรมกล้อง/point-cloud จากนั้น `main.py` จะแปลงไปเป็น
เฟรมฐานหุ่นยนต์

### `extract_3d_data(target_pixels, target_names, show_3d=True)` ทีละขั้น
1. ดึงเฟรม RGB-D ที่ align แล้วใหม่จากกล้อง (`get_raw_frame`)
2. สร้าง Open3D point cloud จากภาพ RGB-D โดยใช้ intrinsics ของ RealSense แล้ว
   `transform([...พลิก Y และ Z...])` เพื่อให้โลกเป็น right-handed (ขึ้น = +Y, ไปข้างหน้า = −Z)
3. ประมาณ **normal** ต่อจุด (`KDTreeSearchParamHybrid(radius=0.01, max_nn=30)`)
4. สำหรับแต่ละพิกเซลเป้าหมาย `(u, v)`:
   - **กู้ค่า depth ที่เป็นศูนย์ (แก้ปัญหาทองแดงสะท้อนแสง):** ถ้า depth ที่พิกเซลเป็น 0 ให้ขยาย
     รัศมีค้นหาจาก 2→7 px แล้วเฉลี่ยเพื่อนบ้านที่ไม่เป็นศูนย์ ถ้าในหน้าต่าง 15×15 ยังเป็นศูนย์ทั้งหมด
     จะ **ข้าม** เป้าหมายนี้ (รายงานขึ้นไปเป็น `no_targets`)
   - **Back-project** ด้วย intrinsics:
     ```
     Z = depth_raw * depth_scale
     X = (u - cx) * Z / fx
     Y = (v - cy) * Z / fy
     ```
   - หาจุดใน point cloud ที่ใกล้ที่สุด แล้วใช้ **normal ของมันเป็นแกน Z เฉพาะที่** สร้างแกน X และ Y
     ให้ตั้งฉาก (โดยใช้ค่าตั้งต้นที่เสถียรเพื่อเลี่ยง degeneracy) → ได้เมทริกซ์การหมุน 3×3
   - แปลงเป็น Euler `(roll, pitch, yaw)`
   - เก็บ: `[X, -Y, -Z, roll, pitch, yaw, rotation_matrix]`
     **สังเกตว่า index 6 = เมทริกซ์ 3×3 เต็ม** — `main.py` ใช้ตัวนี้ (ไม่ใช่ค่า Euler)
     ในการแปลงทิศทางไปเฟรมฐาน ส่วนค่า Euler มีไว้สำหรับ logging
5. แคช cloud + ลูกบอลต่อเป้าหมาย + แกนพิกัด ไว้ใน `self._last_geometries` เพื่อให้ผู้ใช้เปิด
   3D viewer ภายหลังได้ (ปุ่ม `p`) โดยไม่ต้องรันการตรวจจับใหม่

### เมธอดอื่น ๆ
- `show_collected_3d(...)` — เปิด Open3D viewer บน geometry ที่แคชไว้ (บล็อกจนกว่าจะปิดหน้าต่าง)
  ผูกกับปุ่ม `p` ใน `main.py`
- `re_express_in_marker_frame(rvec, tvec)` — ย้ายจุดศูนย์กลางของฉากที่แคชไว้ทั้งหมดให้อยู่บน
  ArUco marker (เครื่องมือ debug เพื่อตรวจสอบการแปลงด้วยสายตา) ไม่ได้ใช้ในเส้นทางตอนรันจริง

### เฟรม — ส่วนที่ทำให้ทุกคนสับสน
มีสามเฟรมในระบบ ต้องแยกให้ออก:

| เฟรม | +X | +Y | +Z | ใช้โดย |
|------|----|----|----|--------|
| OpenCV camera | ขวา | ลง | ไปข้างหน้า | back-projection, hand-eye |
| Point-cloud (Open3D, พลิกแล้ว) | ขวา | ขึ้น | ถอยหลัง | ค่าที่ `extract_3d_data` คืน |
| Robot base (KUKA) | — | — | — | สิ่งที่ PLC ต้องการ |

`extract_3d_data` คืนค่าในเฟรม **point-cloud** (Y/Z พลิกแล้ว) จากนั้น `encode_target_pose`
ของ `main.py` จะพลิก Y และ Z กลับเป็นเฟรม **camera** แล้วใช้การแปลง `H_cam2base` เพื่อไปถึง
เฟรม **base** ดู [main-loop.th.md](main-loop.th.md)
