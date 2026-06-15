# 3. Communication — Camera & PLC

> 🌐 Language: **English** | [ไทย](communication.th.md)

The `communication/` package is the bridge to hardware:

- [`communication/realsense.py`](../../communication/realsense.py) — the camera.
- [`communication/plc_comm.py`](../../communication/plc_comm.py) — the PLC.

---

## 3.1 `realsense.py` — `DepthCamera`

Thin wrapper around `pyrealsense2`, configured specifically for **shiny, specular
metal surfaces** (which are hard for depth cameras).

Key setup in `__init__`:
- Reads `depth_scale` from the device (raw depth units → meters).
- **`visual_preset = 4` (High Density)** — better coverage on low-texture / specular metal.
- Three depth **filters** applied to every frame: `spatial → temporal → hole_filling(2)`.
- **Aligns depth to the color stream** so pixel `(u, v)` in RGB maps to the same `(u, v)`
  in depth — essential for the 2D→3D step.
- Streams depth (`z16`) + color (`bgr8`) at the configured resolution, 30 FPS.

Methods:
| Method | Returns | Used by |
|--------|---------|---------|
| `get_frame()` | `(ok, depth_np, color_np)` numpy arrays | quick previews |
| `get_raw_frame()` | `(ok, depth_frame, color_frame)` RealSense frame objects | detection + point cloud (keeps intrinsics) |
| `get_color_intrinsics()` | `(camera_matrix, dist_coeffs)` | calibration (`board_detect`) |
| `release()` | — | shutdown |

Both `get_*frame` methods apply the depth filters. `get_raw_frame` returns the native
frame objects because the transformer needs `.profile...intrinsics` and `depth_scale`.

---

## 3.2 `plc_comm.py` — `PLCCommunicator`

Wraps `pymcprotocol.Type3E` (binary) with the robustness the factory floor needs.

### What it gives you
- **Auto-reconnect with cooldown.** Every read/write goes through `_call()`, which on
  failure tries to reconnect **once** (rate-limited to one attempt per 2 s) and retries.
  Transient TCP drops don't crash the loop.
- **Safe defaults on read failure.** `read_word` → `0`, `read_bit` → `[0]`, etc. So a
  dropped link reads as "nothing happening" rather than throwing.
- **Thread lock.** All PLC access is serialized (the heartbeat runs on its own thread).
- **int16 clamping.** Writes are clamped to `[-32768, 32767]` so an out-of-range value
  can't corrupt the packet.

### API
| Method | Purpose |
|--------|---------|
| `connect()` / `disconnect()` | open/close session |
| `read_bit(d)` / `read_bits(d, n)` | read M/X bits |
| `read_word(d)` / `read_words(d, n)` | read D words |
| `write_bit(d, v)` / `write_bits(d, [..])` | write bits (block write is one packet) |
| `write_word(d, v)` / `write_words(d, [..])` | write words |
| `write_scaled_word(d, f, mul)` | `write_word(d, round(f*mul))` |
| `write_slot(base, idx, words_per_slot, values)` | write one result slot at `base + idx*words_per_slot` |
| `start_heartbeat(d, interval)` / `stop_heartbeat()` | background counter thread |

`write_bits` writing 4 consecutive M devices in **one** packet is how the four status bits
(ready/error/busy/complete) are pushed together — see `set_status` in `main.py`.

Helper: `_offset_device("D1003", 4)` → `"D1007"` (used by `write_slot`).

---

## 3.3 PLC register map (the whole contract)

Everything the PLC and PC agree on lives in the `plc:` block of `config.yaml`. This is the
**first place to look** for any "not communicating" problem.

### PLC → PC (inputs)
| Device | Type | Meaning |
|--------|------|---------|
| `M1500` `trigger_device` | bit | PLC raises = "scan now" |
| `D1100` `program_no_device` | word | which program/model to run |
| `D2000` `pose_device` (12 words) | 6×int32 | **live robot pose** X Y Z A B C (mm/deg ×1000); read every trigger |

### PC → PLC (status outputs)
| Device | Type | Meaning |
|--------|------|---------|
| `D1000` `heartbeat_device` | word | rolling counter, +1/sec — PLC uses it to detect a dead PC |
| `D1001` `error_code_device` | word | last error code (table below) |
| `M1000` `status_ready_device` | bit | idle, ready for next trigger |
| `M1001` `status_error_device` | bit | error |
| `M1002` `status_busy_device` | bit | scan in progress |
| `M1003` `status_complete_device` | bit | results written, awaiting ack |

> The four status bits are mapped to **consecutive M devices starting at `M1000`** so they
> can be written in one packet. If you move them, keep them consecutive.

### PC → PLC (results)
| Device | Type | Meaning |
|--------|------|---------|
| `D1002` `amount_device` | word | number of valid points this cycle |
| `D1003` `slot_base_device` | words | start of the per-point result slots |

**Slot layout** (current `main.py`): **14 words per slot**, each value an `int32`
(low-word-first, ×1000, mm/deg):
```
slot k starts at D1003 + k*14
  +0  X    +2  Y    +4  Z
  +6  A    +8  B    +10 C
  +12 Conf (×100)
```
Up to `max_points = 5` slots.

> Note: `config.yaml` now sets `words_per_slot: 14` to match this layout, but the value
> is documentation-only — `main.py` hard-codes `words_per_slot = 14` (see line ~358) and
> sends the full 6-DOF pose as int32. The source of truth is the code:
> **14 words, int32, X Y Z A B C Conf.**

### Error codes (`D1001`)
| Code | Meaning |
|------|---------|
| 0 | OK |
| 1 | invalid program no. |
| 2 | no targets found |
| 3 | camera read failure |
| 99 | internal error |

### Calibration-only handshake (used by `aruco_calibate.py`, not the runtime)
| Device | Meaning |
|--------|---------|
| `M2000` `calib_trigger_device` | PLC sets =1 when the KUKA has reached the calibration pose |
| `M2001` `calib_ack_device` | PC sets =1 after recording that pose |

### Debug-only test registers (written from `--debug` PLC-test mode)
| Device | Meaning |
|--------|---------|
| `D1500` `program_no_test_device` | PC writes a program no. here to test the PLC reading it |

### The pose encoding (important detail)
The robot pose and the result slots both use the **same int32 ×1000, low-word-first**
encoding, decoded/encoded by [`tools/plc_decode.py`](../../tools/plc_decode.py)
(`decode_pose` / `encode_pose`). If the PLC's KUKA REALs come back **high-word-first**, set
`pose_word_swap: true` in config — that flips lo/hi in the decoder. See [tools.md](tools.md).

---

## 3.4 The handshake (one trigger cycle)

```
PLC                                   PC (main.py)
 write D1100 = program_no   ──►  polls D1100 (sticky: remembers last valid)
 set   M1500 = 1 (trigger)  ──►  reads M1500
                                 status: ready=0, busy=1
                                 read live pose from D2000, rebuild Cam→Base
                                 capture frame, detect, lift to 3D, transform
                                 write D1002 = amount
                                 write D1003.. = slots (14 words each)
                                 status: busy=0, complete=1
                            ◄──  waits for M1500 = 0  (PLC ack)
 clear M1500 = 0            ──►  holds complete=1 for complete_pulse_sec (1 s)
                                 status: ready=1, complete=0
```

The `complete=1` hold (`complete_pulse_sec`) exists so a slow HMI scan can still latch the
signal. Polling of the PLC is throttled to `poll_interval_sec` (0.1 s) so we don't saturate
the link at camera frame rate.
