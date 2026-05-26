# Solution Methodology

End-to-end methodology of the Heat Exchanger Vision System: from PLC trigger to
3D pose written back into PLC registers. The system pairs a 2D template-matching
detector with depth from an Intel RealSense camera to produce 6-DoF target poses
for the PLC/robot.

---

## 1. System Overview

```
   PLC  ──trigger / program no.──►  PC (Vision)  ──X,Y,Z,Conf──►  PLC
    ▲                                   │
    │                                   ▼
    └────── status / heartbeat ──── RealSense D4xx (RGB + Depth)
```

Three cooperating layers:

| Layer            | Module                          | Responsibility                                  |
|------------------|---------------------------------|-------------------------------------------------|
| Acquisition      | [communication/realsense.py](../communication/realsense.py) | Aligned RGB-D frames, depth filtering            |
| Perception       | [core/detector.py](../core/detector.py), [core/transformer.py](../core/transformer.py) | 2D feature matching → 3D pose                    |
| Integration      | [communication/plc_comm.py](../communication/plc_comm.py), [main.py](../main.py) | MC-protocol I/O, handshake, status, error codes  |

Runtime configuration lives in [config.yaml](../config.yaml) (camera, programs,
PLC device addresses, scaling).

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
camera so that depth is usable on the shiny copper of heat-exchanger plates:

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
8. Return `{target_name: [X, Y, -Y_flipped, -Z_flipped, roll, pitch, yaw]}`.

The point cloud, target spheres, and coordinate axes are cached on the
transformer (`_last_geometries`) so the operator can open an Open3D viewer with
the `p` key without rerunning detection.

---

## 6. PLC Integration

### 6.1 Protocol

[communication/plc_comm.py](../communication/plc_comm.py) speaks **MELSEC
MC-protocol (Type3E binary)** over TCP at the address in
[config.yaml:20-21](../config.yaml#L20-L21). All access is wrapped under a
thread lock with a rate-limited reconnect (2 s cooldown) so transient drops do
not crash the loop.

### 6.2 Handshake (one trigger cycle)

```
PLC                                  PC
 ─ write D1100 = program_no   ──►  poll (D1100)        sticky
 ─ set  M1500 = 1 (trigger)   ──►  read  (M1500)
                                   set status: ready=0, busy=1
                                   capture frame, detect, lift to 3D
                                   write D1002 = amount
                                   write D1003.. = X,Y,Z,Conf per slot
                                   set status: busy=0, complete=1
                              ◄──   wait for M1500 = 0 (PLC ack)
 ─ clear M1500 = 0           ──►  hold complete=1 for COMPLETE_PULSE_SEC
                                   set status: ready=1, complete=0
```

The `complete=1` pulse is held for 1 second
([main.py:23-24](../main.py#L23-L24)) so a slow HMI scan can still latch it.

### 6.3 Status Packing (latency optimization)

The four status bits (`ready, error, busy, complete`) are mapped to four
consecutive M devices and pushed in **one** `write_bits` packet rather than
four separate writes ([main.py:56-72](../main.py#L56-L72)). A small in-process
cache (`_last_status`) makes partial updates possible without an extra read.

### 6.4 Scaling Real-World Units → INT16

PLC words are 16-bit. Floats are encoded by integer scaling
([main.py:98-100, 321-330](../main.py#L98-L100)):

| Value      | Multiplier | Encoded as           |
|------------|------------|----------------------|
| X / Y / Z  | × 10000    | meters → 0.1 mm units |
| Confidence | × 100      | percent × 100        |

Slot layout from `D1003` (`slot_base_device`):

```
D1003..D1006  → slot 1 (X, Y, Z, Conf)
D1007..D1010  → slot 2
...           up to max_points = 5
```

### 6.5 Error Codes (`error_code_device`, `D1001`)

| Code | Meaning              |
|------|----------------------|
| 0    | OK                   |
| 1    | Invalid program no.  |
| 2    | No targets found     |
| 3    | Camera read failure  |
| 99   | Internal error       |

### 6.6 Heartbeat

A background thread increments a counter and writes it to `D1000` every second
([main.py:92](../main.py#L92)) so the PLC can detect a dead PC.

---

## 7. Main Loop (orchestration)

[main.py:118-350](../main.py#L118-L350) is the cycle-driver. Per iteration:

1. Pull a fresh RGB-D frame from the camera.
2. **Throttled PLC poll** (10 Hz by default, independent of camera FPS) reads
   the program number and trigger bit. Polling at camera FPS would saturate
   the PLC link.
3. Render a live preview with the active program's detections (bounding box +
   target dot + label per *best* point).
4. On trigger:
   - Lock `current_program_no`, set `busy=1`.
   - Re-capture a frame for the actual scan (a still frame is more reliable
     than the preview one used for visualization).
   - Run detector → best-per-point filter → 3D lift.
   - Show a "trigger result" grid with one tile per sub-template, BEST tiles
     framed in green ([main.py:362-421](../main.py#L362-L421)).
   - Write `amount`, then per-slot `(X, Y, Z, Conf)`.
   - Persist a JSON snapshot to `data/logs/position_mem.json` for traceability.
   - Pulse `complete=1`, wait for PLC to clear the trigger, then return to
     `ready=1`.

### 7.1 Debug Mode (`--debug`)

Adds keyboard control to operate without a PLC:

- `1`–`9`: select program manually.
- `t`: manual trigger.
- `b`: enter PLC-test sub-mode — `1`–`9` writes to `program_no_test_device`
  and `t` pulses `trigger_test_device` so a PLC engineer can verify their side
  of the handshake.
- `p`: open the cached 3D viewer for the last scan.

---

## 8. Failure Modes and Mitigations

| Failure                                   | Where it is handled                                                                                                |
|-------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| PLC TCP drop                              | `_try_reconnect` with 2 s cooldown ([communication/plc_comm.py:57-60](../communication/plc_comm.py#L57-L60))        |
| Stale trigger bit (cache)                 | Trigger consumed locally; loop waits for PLC to clear it before next cycle ([main.py:229-231](../main.py#L229-L231)) |
| Depth = 0 at target pixel (shiny copper)  | Expanding-radius neighborhood mean ([core/transformer.py:56-74](../core/transformer.py#L56-L74))                    |
| Duplicate sub-template matches            | Best-per-point filter ([main.py:43-53](../main.py#L43-L53))                                                         |
| Program not loaded but trigger fires      | `ERR_INVALID_PROGRAM`, status `error=1` ([main.py:238-243](../main.py#L238-L243))                                   |
| Slow HMI scan misses `complete` pulse     | `COMPLETE_PULSE_SEC` hold ([main.py:24](../main.py#L24))                                                            |
| Excessive PLC packet rate                 | `poll_interval_sec` throttle + packed status writes ([main.py:113-116, 56-72](../main.py#L113-L116))                |

---

## 9. End-to-End Data Path (single trigger)

```
PLC trigger  ─►  RGB+Depth frame  ─►  SIFT/FLANN match against ProgramN templates
              ─►  RANSAC homography → target pixel (u,v) + confidence
              ─►  best-per-point filter
              ─►  depth lookup (with 0-recovery) → back-projection → (X,Y,Z)
              ─►  point cloud + normal estimation → 6-DoF pose
              ─►  scale to INT16 → write amount + slot table
              ─►  set complete=1, wait for PLC ack, return to ready
```
