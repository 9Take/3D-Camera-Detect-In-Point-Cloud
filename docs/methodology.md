# Solution Methodology

> 🌐 Language: **English** | [ไทย](methodology.th.md)

End-to-end methodology of the 3D Guidance Robot vision system: from PLC trigger to
a full 6-DoF pose written back into PLC registers. The system pairs a 2D
template-matching detector with depth from an Intel RealSense camera, lifts each
hit to a 3D pose, and transforms it from the **camera frame into the robot base
frame** (via the hand-eye calibration) before sending it to the PLC/robot.

---

## 1. System Overview

```
   PLC  ──trigger / program no. / live robot pose──►  PC (Vision)
    ▲                                                   │  ──X,Y,Z,A,B,C,Conf (base frame)──►  PLC
    │                                                   ▼
    └────────── status / heartbeat ──────────── RealSense D4xx (RGB + Depth)
```

Three cooperating layers:

| Layer            | Module                          | Responsibility                                  |
|------------------|---------------------------------|-------------------------------------------------|
| Acquisition      | [communication/realsense.py](../communication/realsense.py) | Aligned RGB-D frames, depth filtering            |
| Perception       | [core/detector.py](../core/detector.py), [core/transformer.py](../core/transformer.py) | 2D feature matching → 3D pose                    |
| Integration      | [communication/plc_comm.py](../communication/plc_comm.py), [main.py](../main.py) | Cam→Base transform, MC-protocol I/O, handshake, status, error codes  |

Runtime configuration lives in [config.yaml](../config.yaml) (camera, programs,
PLC device addresses, hand-eye calibration, scaling).

> A task-organized handover guide for new engineers lives in
> [handover/](handover/). This file is the deeper, line-referenced reference.

---

## 2. Per-Program Template Library

A "program number" coming from the PLC selects which set of templates to match.
Each program is a folder containing one or more `Point*` subfolders, and each
point can have several sub-templates so that a single physical feature can be
recognized under different viewpoints/lighting:

```
data/templates/ProgramA/
   PointA/
      A1_template.png
      A1_meta.json     # offset_x, offset_y → the exact "click point" inside template
      A2_template.png
      A2_meta.json
   PointB/
      ...
```

Loading rules ([core/detector.py:14-52](../core/detector.py#L14-L52)):

- Only directories starting with `Point` are scanned.
- For each `*_template.png`, SIFT keypoints + descriptors are pre-computed once
  at startup.
- `*_meta.json` (preferred) or `*_offset.txt` defines the *target offset* — the
  pixel inside the template that corresponds to the physical click point. If no
  meta file exists, the template center is used.

Pre-loading one `ObjectDetector` per program ([main.py:32-40](../main.py#L32-L40))
keeps trigger latency low; switching programs at runtime is just a dictionary
lookup.

---

## 3. Frame Acquisition (RealSense)

[communication/realsense.py](../communication/realsense.py) configures the
camera so that depth is usable on shiny, specular metal surfaces:

1. **High-Density visual preset** (`rs.option.visual_preset = 4`) — improves
   coverage on low-texture / specular surfaces.
2. **Depth filters**: spatial → temporal → hole-filling, applied to every raw
   frame in [communication/realsense.py:53-57](../communication/realsense.py#L53-L57).
3. **Alignment to color stream** so that pixel `(u, v)` in the RGB image maps
   directly to a depth value at the same `(u, v)` in the depth image.

Resolution is 640×480 @ 30 FPS for both streams (see
[config.yaml:1-3](../config.yaml#L1-L3)).

---

## 4. 2D Detection (SIFT + FLANN + Homography)

For each frame, [core/detector.py:54-103](../core/detector.py#L54-L103) runs:

1. Convert frame to grayscale; compute SIFT keypoints/descriptors.
2. For every template of the active program:
   - **FLANN k-NN match** (k=2) between template and frame descriptors.
   - **Lowe's ratio test** at 0.7 to keep "good" matches.
   - Require **> 12 good matches** before attempting a pose.
3. **RANSAC homography** `M` from template → frame coordinates.
4. **Confidence** is derived from the number of RANSAC inliers, capped at 100 %:
   `confidence = min(100, inliers / 30 * 100)`.
5. Project the template's offset point through `M` to get the **target pixel**
   `(u, v)` in the live frame; also project the template's four corners to draw
   the bounding polygon.

### 4.1 Best-per-Point Filtering

A `Point` may have several sub-templates that all match. Only the highest-
confidence detection per point is kept ([main.py:43-53](../main.py#L43-L53)),
preventing duplicate writes to the PLC slot table for the same physical
feature.

---

## 5. 2D → 3D Lifting + 6-DoF Pose

[core/transformer.py:21-121](../core/transformer.py#L21-L121) turns each
detected pixel into a 6-DoF pose:

1. **Read depth** at the detected pixel.
2. **Depth recovery** (camera blind-spot fix,
   [core/transformer.py:56-74](../core/transformer.py#L56-L74)): if the depth is
   0, expand a search radius from 2 → 7 px and average non-zero neighbors. If
   no valid depth is found inside that 15×15 window, the target is skipped (and
   reported as ERR_NO_TARGETS upstream).
3. **Back-project** the pixel using the RealSense intrinsics:
   ```
   Z = depth_raw * depth_scale
   X = (u - cx) * Z / fx
   Y = (v - cy) * Z / fy
   ```
4. **Build a point cloud** from the aligned RGB-D image
   ([core/transformer.py:34-42](../core/transformer.py#L34-L42)) and flip Y/Z
   so the world frame is right-handed and "up = +Y, forward = -Z".
5. **Estimate per-point normals** with `KDTreeSearchParamHybrid(0.01, 30)`.
6. **Orient a local frame**: take the surface normal as the local Z axis, pick
   a stable seed for X (`[1,0,0]` or `[0,1,0]` to avoid degeneracy), then
   `Y = Z × X`, `X = Y × Z`.
7. **Convert** the rotation matrix to Euler `(roll, pitch, yaw)` with
   `rotation_matrix_to_euler_angles` in
   [core/transformer.py:6-13](../core/transformer.py#L6-L13).
8. Return `{target_name: [X, -Y, -Z, roll, pitch, yaw, rotation_matrix]}`
   ([core/transformer.py:99](../core/transformer.py#L99)). **Index 6 carries the
   full 3×3 orientation matrix** (in the point-cloud frame); `main.py` uses that
   matrix — not the Euler angles — to transform orientation into the robot base
   frame (§6). The Euler values are kept for logging/templates.

These coordinates are in the **point-cloud frame** (X right, Y up, Z backward).
`main.py` flips Y/Z back to the OpenCV camera frame before the Cam→Base transform.

The point cloud, target spheres, and coordinate axes are cached on the
transformer (`_last_geometries`) so the operator can open an Open3D viewer with
the `p` key without rerunning detection.

---

## 6. Camera → Robot-Base Transform (Hand-Eye)

The detector/transformer produce a pose in the **camera frame**. The robot needs
it in its **base frame**. This conversion is the second half of `main.py` and is
what makes the output directly usable by the KUKA.

### 6.1 The 4×4 chain

`build_cam2base` ([main.py:36-48](../main.py#L36-L48)) composes two transforms
into a single `H_cam2base`:

```
H_cam2base = H_gripper2base @ H_cam2gripper
```

- **`H_cam2gripper`** — the fixed **hand-eye calibration** result
  (`robot.hand_eye_rotation` / `hand_eye_translation` in config). Produced by
  [calibration/aruco_calibate.py](../calibration/aruco_calibate.py); see
  [handover/calibration.md](handover/calibration.md) for how to run/verify it.
- **`H_gripper2base`** — the robot's **photo pose** (where the arm is parked when
  the picture is taken). Built from `(x,y,z,a,b,c)` with `kuka_abc_to_matrix`
  ([main.py:27-33](../main.py#L27-L33)), which uses the KUKA **intrinsic Z-Y-X**
  convention (`Rz(A)·Ry(B)·Rx(C)`).

### 6.2 Live photo pose from the PLC

The photo pose is **read live from the PLC on every trigger**
(`read_robot_scan_pose`, [main.py:51-59](../main.py#L51-L59)) from `pose_device`
(`D2000`, six int32 ×1000, mm/deg), and `H_cam2base` is rebuilt. The static
`robot.scan_pose` in config is only a **fallback** used at startup or if the PLC
read fails / returns all-zero — set it to the true parked pose so a failed read
never sends the robot to the wrong place.

### 6.3 Per-target encoding

`encode_target_pose` ([main.py:75-114](../main.py#L75-L114)) turns one detected
target into its PLC result slot:

1. Flip the point-cloud-frame coords back to the **OpenCV camera frame**
   (`y_cam = -y`, `z_cam = -z`).
2. Add the stick **`ee_offset`** (camera frame, meters) and transform the point
   Cam→Base.
3. Transform the orientation Cam→Base: `R_base = H_cam2base[:3,:3] · F · R_pcd`,
   where `F = diag(1,-1,-1)` undoes the point-cloud flip. Then
   `Rotation.from_matrix(R_base).as_euler("ZYX", degrees=True)` → KUKA `A,B,C`.
4. Encode `X,Y,Z` (mm), `A,B,C` (deg) and Confidence as **int32 ×1000**
   (low-word-first) via [tools/plc_decode.py](../tools/plc_decode.py) → 14 words.

The `ee_offset` only shifts position (so the reported point lands at the stick
tip); orientation is unchanged.

---

## 7. PLC Integration

### 7.1 Protocol

[communication/plc_comm.py](../communication/plc_comm.py) speaks **MELSEC
MC-protocol (Type3E binary)** over TCP at the address in
[config.yaml:36-37](../config.yaml#L36-L37). All access is wrapped under a
thread lock with a rate-limited reconnect (2 s cooldown) so transient drops do
not crash the loop.

### 7.2 Handshake (one trigger cycle)

```
PLC                                  PC
 ─ write D1100 = program_no   ──►  poll (D1100)        sticky
 ─ write D2000 = robot pose   ──►  (read on trigger)
 ─ set  M1500 = 1 (trigger)   ──►  read  (M1500)
                                   set status: ready=0, busy=1
                                   read live pose (D2000), rebuild Cam→Base
                                   capture frame, detect, lift to 3D, transform
                                   write D1002 = amount
                                   write D1003.. = X,Y,Z,A,B,C,Conf per slot
                                   set status: busy=0, complete=1
                              ◄──   wait for M1500 = 0 (PLC ack)
 ─ clear M1500 = 0           ──►  hold complete=1 for COMPLETE_PULSE_SEC
                                   set status: ready=1, complete=0
```

The `complete=1` pulse is held for `complete_pulse_sec` (1 s) so a slow HMI scan
can still latch it.

### 7.3 Status Packing (latency optimization)

The four status bits (`ready, error, busy, complete`) are mapped to four
consecutive M devices (`M1000..M1003`) and pushed in **one** `write_bits` packet
rather than four separate writes (`set_status`,
[main.py:155-168](../main.py#L155-L168)). A small in-process cache
(`_last_status`) makes partial updates possible without an extra read.

### 7.4 Scaling Real-World Units → INT32

Each pose value is sent as a **32-bit integer split into two 16-bit words**
(low-word-first), scaled ×1000 — the same encoding the PLC uses for the robot
pose it sends us, so [tools/plc_decode.py](../tools/plc_decode.py) round-trips
both directions:

| Value          | Scaling | Encoded as            |
|----------------|---------|-----------------------|
| X / Y / Z      | × 1000  | meters → mm (×1000 → µm-resolution int32) |
| A / B / C      | × 1000  | degrees → millidegrees int32 |
| Confidence     | × 100   | percent × 100, int32  |

> Note: `config.yaml` now sets `words_per_slot: 14` to match the present design,
> but the value is documentation-only — `main.py` hard-codes **14 words/slot**
> ([main.py:358](../main.py#L358)) and scales via `plc_decode`. The legacy
> `position_multiplier: 10000` key is unused (the actual scale is ×1000 via
> `POSE_SCALE` in `plc_decode`) and is safe to clean up.

Slot layout from `D1003` (`slot_base_device`), **14 words (7 int32) per slot**:

```
slot k  →  D1003 + k*14
   +0  X    +2  Y    +4  Z
   +6  A    +8  B    +10 C
   +12 Conf
...                          up to max_points = 5
```

### 7.5 Error Codes (`error_code_device`, `D1001`)

| Code | Meaning              |
|------|----------------------|
| 0    | OK                   |
| 1    | Invalid program no.  |
| 2    | No targets found     |
| 3    | Camera read failure  |
| 99   | Internal error       |

### 7.6 Heartbeat

A background thread increments a counter and writes it to `D1000` every second
(`start_heartbeat`,
[communication/plc_comm.py:149-164](../communication/plc_comm.py#L149-L164)) so
the PLC can detect a dead PC.

---

## 8. Main Loop (orchestration)

The `main()` loop ([main.py:336-572](../main.py#L336-L572)) is the cycle-driver.
Per iteration:

1. Pull a fresh RGB-D frame from the camera.
2. **Throttled PLC poll** (10 Hz by default, independent of camera FPS) reads
   the program number and trigger bit. Polling at camera FPS would saturate
   the PLC link.
3. Render a live preview with the active program's detections (bounding box +
   target dot + label per *best* point).
4. On trigger:
   - Lock `current_program_no`, set `busy=1`.
   - **Read the live photo pose from the PLC** and rebuild `H_cam2base` (§6.2);
     fall back to the config pose if unavailable.
   - Re-capture a frame for the actual scan (a still frame is more reliable
     than the preview one used for visualization).
   - Run detector → best-per-point filter → 3D lift.
   - Show a "trigger result" grid with one tile per sub-template, BEST tiles
     framed in green (`_build_trigger_result_grid`,
     [main.py:271-330](../main.py#L271-L330)).
   - Write `amount`, then per-slot `(X, Y, Z, A, B, C, Conf)` after the Cam→Base
     transform (§6.3).
   - Persist a JSON snapshot to `data/logs/position_mem.json` for traceability.
   - Pulse `complete=1`, wait for PLC to clear the trigger, then return to
     `ready=1`.

### 8.1 Debug Mode (`--debug`)

Adds keyboard control to operate without a PLC:

- `1`–`9`: select program manually.
- `t`: manual trigger.
- `b`: enter PLC-test sub-mode — `1`–`9` writes to `program_no_test_device`
  and `t` pulses `trigger_test_device` so a PLC engineer can verify their side
  of the handshake.
- `p`: open the cached 3D viewer for the last scan.

---

## 9. Failure Modes and Mitigations

| Failure                                   | Where it is handled                                                                                                |
|-------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| PLC TCP drop                              | `_try_reconnect` with 2 s cooldown ([communication/plc_comm.py:57-73](../communication/plc_comm.py#L57-L73))        |
| Stale trigger bit (cache)                 | Trigger consumed locally; loop waits for PLC to clear it before next cycle (`_wait_trigger_low`, [main.py:183-192](../main.py#L183-L192)) |
| Depth = 0 at target pixel (shiny copper)  | Expanding-radius neighborhood mean ([core/transformer.py:56-74](../core/transformer.py#L56-L74))                    |
| Robot pose all-zero / PLC read fails      | Fall back to config `scan_pose` for Cam→Base ([main.py:456-465](../main.py#L456-L465))                              |
| Duplicate sub-template matches            | Best-per-point filter (`best_per_point`, [main.py:137-147](../main.py#L137-L147))                                   |
| Program not loaded but trigger fires      | `ERR_INVALID_PROGRAM`, status `error=1` ([main.py:443-448](../main.py#L443-L448))                                   |
| Slow HMI scan misses `complete` pulse     | `complete_pulse_sec` hold ([main.py:561](../main.py#L561))                                                          |
| Excessive PLC packet rate                 | `poll_interval_sec` throttle + packed status writes ([main.py:382-383](../main.py#L382-L383), [155-168](../main.py#L155-L168)) |

---

## 10. End-to-End Data Path (single trigger)

```
PLC trigger  ─►  read live robot photo pose (D2000) → rebuild H_cam2base
              ─►  RGB+Depth frame  ─►  SIFT/FLANN match against ProgramN templates
              ─►  RANSAC homography → target pixel (u,v) + confidence
              ─►  best-per-point filter
              ─►  depth lookup (with 0-recovery) → back-projection → (X,Y,Z)
              ─►  point cloud + normal estimation → 6-DoF pose (camera frame)
              ─►  Cam→Base transform (+ ee_offset) → X,Y,Z,A,B,C (base frame)
              ─►  encode int32 ×1000 → write amount + 14-word slots
              ─►  set complete=1, wait for PLC ack, return to ready
```
