# 3. Communication — กล้องและ PLC

> 🌐 ภาษา: [English](communication.md) | **ไทย**

แพ็กเกจ `communication/` คือสะพานเชื่อมไปยังฮาร์ดแวร์:

- [`communication/realsense.py`](../../communication/realsense.py) — กล้อง
- [`communication/plc_comm.py`](../../communication/plc_comm.py) — PLC

---

## 3.1 `realsense.py` — `DepthCamera`

ตัวห่อหุ้มบาง ๆ รอบ `pyrealsense2` ตั้งค่ามาเพื่อ **พื้นผิวทองแดงสะท้อนแสง** ของแผ่น
heat-exchanger โดยเฉพาะ (ซึ่งยากสำหรับกล้อง depth)

การตั้งค่าหลักใน `__init__`:
- อ่าน `depth_scale` จากอุปกรณ์ (หน่วย depth ดิบ → เมตร)
- **`visual_preset = 4` (High Density)** — ครอบคลุมพื้นผิว texture ต่ำ/สะท้อนแสงได้ดีขึ้น
- ใส่ **ฟิลเตอร์** depth สามตัวกับทุกเฟรม: `spatial → temporal → hole_filling(2)`
- **Align depth เข้ากับ color stream** เพื่อให้พิกเซล `(u, v)` ใน RGB ตรงกับ `(u, v)`
  เดียวกันใน depth — สำคัญมากสำหรับขั้น 2D→3D
- สตรีม depth (`z16`) + color (`bgr8`) ที่ความละเอียดตามตั้งค่า, 30 FPS

เมธอด:
| เมธอด | คืนค่า | ใช้โดย |
|-------|--------|--------|
| `get_frame()` | `(ok, depth_np, color_np)` อาเรย์ numpy | พรีวิวเร็ว ๆ |
| `get_raw_frame()` | `(ok, depth_frame, color_frame)` object ของ RealSense | การตรวจจับ + point cloud (เก็บ intrinsics ไว้) |
| `get_color_intrinsics()` | `(camera_matrix, dist_coeffs)` | การคาลิเบรต (`board_detect`) |
| `release()` | — | ปิดระบบ |

เมธอด `get_*frame` ทั้งสองใส่ฟิลเตอร์ depth ให้ ส่วน `get_raw_frame` คืน object เฟรมดั้งเดิม
เพราะ transformer ต้องใช้ `.profile...intrinsics` และ `depth_scale`

---

## 3.2 `plc_comm.py` — `PLCCommunicator`

ห่อหุ้ม `pymcprotocol.Type3E` (binary) พร้อมความทนทานที่หน้างานโรงงานต้องการ

### ให้อะไรเรา
- **Reconnect อัตโนมัติพร้อม cooldown** ทุกการอ่าน/เขียนผ่าน `_call()` ซึ่งเมื่อ fail จะลอง
  reconnect **หนึ่งครั้ง** (จำกัดอัตราที่หนึ่งครั้งต่อ 2 วินาที) แล้วลองใหม่ การหลุด TCP ชั่วคราว
  จะไม่ทำให้ลูปพัง
- **คืนค่าปลอดภัยเมื่ออ่าน fail** `read_word` → `0`, `read_bit` → `[0]` ฯลฯ ดังนั้นลิงก์ที่หลุด
  จะอ่านได้เป็น "ไม่มีอะไรเกิดขึ้น" แทนที่จะ throw
- **Thread lock** การเข้าถึง PLC ทั้งหมดถูกทำเป็นลำดับ (heartbeat รันบน thread ของตัวเอง)
- **Clamp int16** การเขียนถูกจำกัดอยู่ใน `[-32768, 32767]` เพื่อไม่ให้ค่าที่เกินช่วงทำให้
  แพ็กเก็ตเสีย

### API
| เมธอด | ใช้ทำอะไร |
|-------|-----------|
| `connect()` / `disconnect()` | เปิด/ปิด session |
| `read_bit(d)` / `read_bits(d, n)` | อ่านบิต M/X |
| `read_word(d)` / `read_words(d, n)` | อ่านคำ (word) D |
| `write_bit(d, v)` / `write_bits(d, [..])` | เขียนบิต (block write เป็นหนึ่งแพ็กเก็ต) |
| `write_word(d, v)` / `write_words(d, [..])` | เขียนคำ |
| `write_scaled_word(d, f, mul)` | `write_word(d, round(f*mul))` |
| `write_slot(base, idx, words_per_slot, values)` | เขียนหนึ่ง slot ที่ `base + idx*words_per_slot` |
| `start_heartbeat(d, interval)` / `stop_heartbeat()` | thread นับ counter เบื้องหลัง |

การที่ `write_bits` เขียนอุปกรณ์ M ติดกัน 4 ตัวใน **หนึ่ง** แพ็กเก็ต คือวิธีที่บิตสถานะทั้งสี่
(ready/error/busy/complete) ถูกส่งพร้อมกัน — ดู `set_status` ใน `main.py`

ตัวช่วย: `_offset_device("D1003", 4)` → `"D1007"` (ใช้โดย `write_slot`)

---

## 3.3 ตารางรีจิสเตอร์ PLC (ข้อตกลงทั้งหมด)

ทุกอย่างที่ PLC และ PC ตกลงกันอยู่ในบล็อก `plc:` ของ `config.yaml` นี่คือ **ที่แรกที่ต้องดู**
สำหรับปัญหา "สื่อสารไม่ได้" ใด ๆ

### PLC → PC (ขาเข้า)
| อุปกรณ์ | ชนิด | ความหมาย |
|---------|------|----------|
| `M1500` `trigger_device` | bit | PLC ยกขึ้น = "สแกนเดี๋ยวนี้" |
| `D1100` `program_no_device` | word | จะรันโปรแกรม/โมเดลไหน |
| `D2000` `pose_device` (12 words) | 6×int32 | **pose หุ่นยนต์สด** X Y Z A B C (mm/deg ×1000) อ่านทุก trigger |

### PC → PLC (สถานะขาออก)
| อุปกรณ์ | ชนิด | ความหมาย |
|---------|------|----------|
| `D1000` `heartbeat_device` | word | counter หมุน, +1/วินาที — PLC ใช้ตรวจจับ PC ที่ตายไปแล้ว |
| `D1001` `error_code_device` | word | error code ล่าสุด (ตารางด้านล่าง) |
| `M1000` `status_ready_device` | bit | ว่าง พร้อมรับ trigger ถัดไป |
| `M1001` `status_error_device` | bit | error |
| `M1002` `status_busy_device` | bit | กำลังสแกน |
| `M1003` `status_complete_device` | bit | เขียนผลแล้ว รอ ack |

> บิตสถานะทั้งสี่ map ไปยังอุปกรณ์ M ที่ **ติดกันเริ่มจาก `M1000`** เพื่อให้เขียนได้ในแพ็กเก็ตเดียว
> ถ้าจะย้าย ให้คงความติดกันไว้

### PC → PLC (ผลลัพธ์)
| อุปกรณ์ | ชนิด | ความหมาย |
|---------|------|----------|
| `D1002` `amount_device` | word | จำนวนจุดที่ใช้ได้ในรอบนี้ |
| `D1003` `slot_base_device` | words | จุดเริ่มของ slot ผลลัพธ์ต่อจุด |

**โครงสร้าง slot** (`main.py` ปัจจุบัน): **14 words ต่อ slot** แต่ละค่าเป็น `int32`
(low-word-first, ×1000, mm/deg):
```
slot k เริ่มที่ D1003 + k*14
  +0  X    +2  Y    +4  Z
  +6  A    +8  B    +10 C
  +12 Conf (×100)
```
สูงสุด `max_points = 5` slot

> ⚠️ **ระวัง:** `config.yaml` มี `words_per_slot: 4` แต่ `main.py` override เป็น **14**
> ภายในโค้ด (`words_per_slot = 14`, ดูบรรทัด ~358) เพราะตอนนี้ส่ง 6-DOF เต็มเป็น int32
> ไม่ใช่แค่ X/Y/Z/Conf เป็น int16 คำอธิบาย 4-word ใน `methodology.md` มาจากดีไซน์เก่า
> เชื่อโค้ด: **14 words, int32, X Y Z A B C Conf**

### Error codes (`D1001`)
| Code | ความหมาย |
|------|----------|
| 0 | OK |
| 1 | หมายเลขโปรแกรมไม่ถูกต้อง |
| 2 | ไม่พบเป้าหมาย |
| 3 | อ่านกล้อง fail |
| 99 | error ภายใน |

### Handshake เฉพาะการคาลิเบรต (ใช้โดย `aruco_calibate.py` ไม่ใช่ตอนรันจริง)
| อุปกรณ์ | ความหมาย |
|---------|----------|
| `M2000` `calib_trigger_device` | PLC ตั้ง =1 เมื่อ KUKA ไปถึง pose คาลิเบรตแล้ว |
| `M2001` `calib_ack_device` | PC ตั้ง =1 หลังบันทึก pose นั้น |

### รีจิสเตอร์ทดสอบเฉพาะ debug (เขียนจากโหมด PLC-test ใน `--debug`)
| อุปกรณ์ | ความหมาย |
|---------|----------|
| `D1500` `program_no_test_device` | PC เขียนหมายเลขโปรแกรมที่นี่ เพื่อทดสอบว่า PLC อ่านได้ |

### การเข้ารหัส pose (รายละเอียดสำคัญ)
ทั้ง pose ของหุ่นยนต์และ slot ผลลัพธ์ใช้การเข้ารหัสแบบเดียวกัน คือ **int32 ×1000, low-word-first**
ถอด/เข้ารหัสด้วย [`tools/plc_decode.py`](../../tools/plc_decode.py) (`decode_pose` / `encode_pose`)
ถ้า KUKA REAL ของ PLC ส่งกลับมาเป็น **high-word-first** ให้ตั้ง `pose_word_swap: true` ใน config
— ตัวนั้นจะสลับ lo/hi ในตัวถอดรหัส ดู [tools.th.md](tools.th.md)

---

## 3.4 Handshake (หนึ่งรอบ trigger)

```
PLC                                   PC (main.py)
 write D1100 = program_no   ──►  poll D1100 (sticky: จำค่าที่ใช้ได้ล่าสุด)
 set   M1500 = 1 (trigger)  ──►  read M1500
                                 status: ready=0, busy=1
                                 อ่าน pose สดจาก D2000, สร้าง Cam→Base ใหม่
                                 จับเฟรม, ตรวจจับ, ยกเป็น 3D, แปลง
                                 write D1002 = amount
                                 write D1003.. = slot (14 words ต่อ slot)
                                 status: busy=0, complete=1
                            ◄──  รอ M1500 = 0  (PLC ack)
 clear M1500 = 0            ──►  ค้าง complete=1 เป็นเวลา complete_pulse_sec (1 วินาที)
                                 status: ready=1, complete=0
```

การค้าง `complete=1` (`complete_pulse_sec`) มีไว้เพื่อให้ HMI ที่ scan ช้ายัง latch สัญญาณได้ทัน
การ poll PLC ถูกจำกัดที่ `poll_interval_sec` (0.1 วินาที) เพื่อไม่ให้ลิงก์ล้นที่อัตราเฟรมของกล้อง
