# 6. Configuration — `config.yaml`

> 🌐 Language: **English** | [ไทย](configuration.th.md)

Every runtime setting lives in [`config.yaml`](../../config.yaml). No constants are
hard-coded in `main.py` that you'd normally need to change (a couple of detector tuning
values are in `core/detector.py` — see [core.md](core.md)). This page explains every block.

---

## `camera:`
```yaml
camera:
  resolution_width: 640
  resolution_height: 480
```
RealSense stream size for both color and depth. 640×480 @ 30 FPS is the tested config.

---

## `charuco:` — calibration board geometry
```yaml
charuco:
  squares_x: 7          # squares across
  squares_y: 5          # squares down
  square_length: 0.0494 # meters, chessboard square side
  marker_length: 0.025  # meters, ArUco marker side
  dictionary: DICT_6X6_250
```
**Must match your physical printed board exactly.** Used only by calibration. (Note: a 7×5
board has 6×4 = 24 *corners* — don't confuse corner ids with marker ids.)

---

## `calibration:` — hand-eye capture settings
```yaml
calibration:
  total_poses: 16          # poses to collect before solving
  min_charuco_corners: 8   # min corners for a stable board pose
  max_reproj_px: 2.0       # reject board poses worse than this (kills 180° flips)
  settle_sec: 0.5          # wait after trigger before recording (robot settle)
  pose_stable_tol: 0.5     # max change (mm/deg) between two consecutive pose reads
  debug_raw_pose: true     # print raw words + both decodings each capture
  debug_handshake: true    # print trigger/ack bit + state changes
```
See [calibration.md](calibration.md).

---

## `programs:` — the template library selector
```yaml
programs:
  1: { name: ProgramA, template_dir: data/templates/ProgramA }
  2: { name: ProgramB, template_dir: data/templates/ProgramB }
  ...
```
The number is what the PLC sends in `program_no_device`. `main.py` pre-loads one
`ObjectDetector` per program at startup so switching at runtime is instant. Each
`template_dir` contains `Point*/` subfolders of taught templates.

---

## `plc:` — the PLC contract
This is the big one. See [communication.md](communication.md) §3.3 for the full register
map and what each device means. Highlights:

```yaml
plc:
  ip: 192.168.1.165
  port: 5010

  trigger_device: M1500       # PLC→PC: start scan
  program_no_device: D1100    # PLC→PC: which program
  use_live_scan_pose: false   # true = read live robot pose from PLC every trigger;
                              # false = always use the manual robot.scan_pose below
  pose_device: D2000          # PLC→PC: live robot pose (6× int32)
  pose_word_count: 12
  pose_word_swap: false       # set true if KUKA REALs come back high-word-first

  calib_trigger_device: M2000 # calibration handshake only
  calib_ack_device: M2001

  heartbeat_device: D1000     # PC→PLC status
  error_code_device: D1001
  status_ready_device: M1000  # ready/error/busy/complete must stay CONSECUTIVE
  status_error_device: M1001
  status_busy_device: M1002
  status_complete_device: M1003

  amount_device: D1002        # PC→PLC results
  slot_base_device: D1003
  words_per_slot: 14          # 7 int32: X Y Z A B C Conf (matches main.py)
  max_points: 5

  heartbeat_interval_sec: 1.0
  poll_interval_sec: 0.1      # how often to poll PLC trigger/program
  complete_pulse_sec: 1.0     # hold complete=1 this long after ack
  position_multiplier: 10000
  confidence_multiplier: 100
  error_codes: { ok: 0, invalid_program: 1, no_targets: 2, camera: 3, internal: 99 }
```

> Note: `words_per_slot` is now `14` to match `main.py`, but the value is
> documentation-only — `main.py` hard-codes **14 words/slot** and sends full 6-DOF
> (X Y Z A B C Conf) as int32. The `position_multiplier` (10000) is a leftover from the
> old int16 path and is unused — the current code scales pose by ×1000 via
> `tools/plc_decode.py`. Trust the code: **14 words/slot, int32 ×1000.**

---

## `robot:` — calibration result + geometry
```yaml
robot:
  hand_eye_rotation:    [...3x3...]   # R_cam2gripper from calibration (unitless)
  hand_eye_translation: [...3...]     # t_cam2gripper, METERS (calib prints mm → ÷1000)

  scan_pose:                          # photo pose (SmartPAD WORLD frame)
    x: 0.530  # m                     # used as-is when use_live_scan_pose is false.
    y: 0.015  # m                     # When live mode is on it is the startup default
    z: 0.090  # m                     # and the fallback if the PLC read fails / is
    a: -90.0  # deg                   # all-zero. Set it to the real parked photo pose.
    b: 0.0
    c: -180.0

  ee_offset:                          # stick end-effector tip offset from the CAMERA,
    x: 0.0   # m                      # measured in the camera frame (X right, Y down,
    y: 0.0   # m                      # Z forward). Added to each detected point so the
    z: -0.295 # m                     # reported X/Y/Z lands at the stick tip. Orientation
                                      # is NOT changed by this.
```
- **`hand_eye_*`** — from running the calibration; the only manual handoff. See
  [calibration.md](calibration.md).
- **`use_live_scan_pose`** — selects where the photo pose comes from. `false` (default):
  always use the manual `scan_pose` below. `true`: `main.py` reads the *live* pose from the
  PLC (`pose_device`) on every trigger and rebuilds Cam→BASE from it.
- **`scan_pose`** — the manual photo pose. Used as-is when `use_live_scan_pose` is `false`;
  in live mode it is the startup default and the fallback if the PLC read fails / is all-zero.
- **`ee_offset`** — shifts the reported point from the camera optical center to the actual
  tool tip. Adjust if the robot consistently lands offset along one axis.

---

## `paths:`
```yaml
paths:
  debug_dir: data/templates/debug   # where create_template --debug saves .ply files
  save_dir: data/logs
  position_mem: data/logs/position_mem.json   # last scan's full result (written each cycle)
```
