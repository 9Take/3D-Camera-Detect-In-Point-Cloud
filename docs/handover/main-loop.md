# 7. Main Loop — How It All Fits Together

> 🌐 Language: **English** | [ไทย](main-loop.th.md)

[`main.py`](../../main.py) is the orchestrator. If you understand this file, you understand
the system. Everything else is a module it calls. Read this with `main.py` open beside it.

---

## Startup (`main()` → `setup_systems()`)

1. `load_config()` reads `config.yaml`.
2. `setup_systems()`:
   - opens the `DepthCamera`,
   - `load_detectors()` — builds one `ObjectDetector` per program (templates pre-loaded),
   - builds the `PointCloudTransformer`,
   - connects the `PLCCommunicator` and starts the **heartbeat thread**,
   - pushes the initial PLC state: `error=ok`, `ready=1, busy=0, complete=0, error=0`.
3. Builds the default `H_CAM2BASE` from `build_cam2base(robot_cfg, scan_pose)` — the camera→
   base transform using the *config* scan pose (refreshed live each trigger only when
   `plc.use_live_scan_pose` is true).

---

## The forever loop (per iteration)

```
grab RGB-D frame
  └─ (throttled to poll_interval_sec) read program_no from PLC → set current_program_no
render_live_view()  ── live window: best detection per point + key hints + console heartbeat
handle keyboard     ── q/ESC quit · p 3D view · (debug) 1-9 program · t trigger · b PLC-test
  └─ (throttled) read trigger bit
if not triggered: continue
```

When a trigger fires (PLC bit = 1, or `t` in debug) → run **one scan cycle**.

---

## One scan cycle (the important part)

```
1. set_status(ready=0, busy=1, complete=0, error=0); error_code = ok
2. validate program_no  ── if invalid → error_code=invalid_program, error=1,
                           wait for trigger to clear, abort cycle
3. (only if plc.use_live_scan_pose) read_robot_scan_pose(plc) from D2000  ── live robot pose
     └─ if valid: rebuild H_CAM2BASE from it
     └─ if all-zero/fail: keep the config-based H_CAM2BASE (with a warning)
     └─ if use_live_scan_pose is false: skip entirely, keep config H_CAM2BASE
4. capture a fresh scan frame  ── if camera fails → error_code=camera, abort cycle
5. detector.detect(scan_frame)  ── all template hits
6. best_per_point(...)          ── keep only the highest-confidence template per Point
7. show "Trigger Result" grid (best per point framed green)
8. if no targets → report_no_results() (amount=0, error=no_targets, pulse complete) → next
9. transformer.extract_3d_data(best pixels)  ── 6-DOF per target in point-cloud frame
     └─ if nothing liftable (all depth-zero) → report_no_results() → next
10. write amount_device = number of targets (capped at max_points)
11. for each target:  encode_target_pose(...) → write_slot(...)
12. dump data/logs/position_mem.json  (full result snapshot)
13. set_status(busy=0, complete=1); wait for PLC to clear trigger;
    hold complete for complete_pulse_sec; set_status(ready=1, complete=0)
```

---

## `encode_target_pose()` — the frame transform, in detail

This is the mathematical heart. For one target it does:

1. **Input:** the transformer's output `[X, -Y, -Z, roll, pitch, yaw, R_pcd]` (point-cloud
   frame). It uses position `[0:3]` and the **3×3 matrix at index 6** — not the Euler angles.
2. **Flip back to OpenCV camera frame:** `y_cam = -y_trans`, `z_cam = -z_trans`
   (point-cloud frame had Y up / Z back; camera frame is Y down / Z forward).
3. **Add the stick `ee_offset`** (in camera frame) and transform the point Cam→Base:
   `cam_point_to_base(H_cam2base, …)`.
4. **Transform orientation** Cam→Base: `R_base = H_cam2base[:3,:3] @ F @ R_pcd`, where
   `F = diag(1,-1,-1)` flips the point-cloud frame back to raw camera frame.
5. **To KUKA angles:** `Rotation.from_matrix(R_base).as_euler("ZYX", degrees=True)` →
   `A, B, C` (matching the KUKA intrinsic Z-Y-X convention used in calibration).
6. **Encode** X,Y,Z (mm), A,B,C (deg), Conf as int32 ×1000 → **14 words** via
   `encode_pose` + `int32_to_words`.

Returns `(slot_words, (x,y,z) meters, (A,B,C) degrees)` — the tuples are for logging and the
position-memory JSON.

> The two frame helpers `kuka_abc_to_matrix` / `build_cam2base` live at the top of
> `main.py`. `kuka_abc_to_matrix` uses the same Z-Y-X convention as `tools/geometry.py`'s
> `rotation_matrix_from_abc` — keep them consistent if you ever change one.

---

## Status & error helpers (top of `main.py`)

- `set_status(plc, cfg, ready=…, busy=…, complete=…, error=…)` — updates only the bits you
  pass, remembers the rest in `_last_status`, and pushes all four to the PLC in **one**
  `write_bits` packet. Use this for every status change.
- `report_no_results(...)` — the standard "trigger produced nothing" handshake: amount=0,
  no_targets error, pulse complete, wait for trigger to clear, return to ready.
- `_wait_trigger_low(...)` — blocks until the PLC clears the trigger bit (its ack), with a
  10 s timeout.

---

## Shutdown

On `q`/`ESC`/Ctrl-C the `finally` block sets `error=1` (so the PLC knows the PC is down),
disconnects the PLC (which stops the heartbeat), releases the camera, and closes windows.

---

## Quick mental model

> **The loop is: poll PLC → on trigger, take a picture, find the points, turn each into a
> robot-base 6-DOF pose, write them to PLC registers, handshake done.** Everything in
> `core/`, `communication/`, and the geometry helpers is a tool this loop calls.
