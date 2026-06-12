# 4. Calibration — Hand-Eye

This is the most error-prone part of the whole system. Read this fully before running it.

**File:** [`calibration/aruco_calibate.py`](../../calibration/aruco_calibate.py)
**Helpers:** [`tools/board_detect.py`](../../tools/board_detect.py),
[`tools/geometry.py`](../../tools/geometry.py),
[`tools/plc_decode.py`](../../tools/plc_decode.py)

---

## Why we calibrate

The camera is mounted on the robot wrist ("eye-in-hand"). To send the robot to a point we
*see*, we must know the fixed transform from the **camera frame** to the **gripper/TCP
frame** — the `R_cam2gripper`, `t_cam2gripper` pair. That's the hand-eye calibration.
Combined with the robot's known gripper→base pose, it lets us convert any camera-frame
point into a robot-base coordinate.

The result lives in `config.yaml` under `robot.hand_eye_rotation` /
`robot.hand_eye_translation`, and `main.py`'s `build_cam2base()` combines it with the photo
pose to get the full `H_cam2base` 4×4 matrix.

---

## The method in one picture

We show a **ChArUco board** (a chessboard with ArUco markers) fixed in the world. We move
the robot to ~16 different poses. At each pose we record two things:
- the **robot pose** (gripper→base), read from the PLC,
- the **board pose in the camera** (target→cam), from `detect_board_pose`.

OpenCV's `cv2.calibrateHandEye(...)` solves for the camera→gripper transform from those
pairs (the classic `AX = XB` problem).

```
For each of N poses:
  robot at pose i ──► PLC raises M2000 ──► PC sees board, reads robot pose ──► PC raises M2001
                                                                              ──► PLC moves to next
Collect N pairs → cv2.calibrateHandEye(method=PARK) → R/t cam→gripper
```

---

## ⚠️ Two non-negotiable gotchas (both cost days — see `memory/`)

1. **OpenCV 4.7.0 `CharucoBoard.matchImagePoints()` is broken** — it returns the wrong
   (marker) corners and scrambles the obj↔img pairing, giving ~1400 px reprojection error
   even on a *perfect* rendered board. `board_detect.py` bypasses it and builds
   correspondences directly from `getChessboardCorners()[ids]`, dropping reproj to ~0.16 px.
   **Do not revert this.** (`memory/opencv-470-matchimagepoints-bug.md`)

2. **Use `PARK`, not `TSAI`.** Our poses have large inter-pose rotations (148°/173°/…),
   where TSAI's rotation solver is fragile — it gave 351–560 mm residual where PARK gave
   29 mm on the *same good data*. `aruco_calibate.py` runs all 5 methods for comparison but
   keeps `solved["PARK"]`. (`memory/hand-eye-use-park-not-tsai.md`)

---

## The 4-phase PLC handshake

Capture is driven by the robot, interlocked so a pose can never be recorded mid-move or
missed:

```
state: wait_trigger
  PLC sets M2000=1 (KUKA reached pose)
    → PC waits SETTLE_SEC for the arm to settle + pose words to stabilize
    → PC reads board pose + robot pose (twice, must match within POSE_STABLE_TOL)
    → PC records the pair, sets M2001=1 ("camera complete ok")
state: wait_release
  PLC sees M2001=1, drops M2000=0
    → PC sees M2000=0, drops M2001=0 → back to wait_trigger for next pose
```

`M2001` is **held high** until the PLC drops `M2000`, so the ack can't be missed.

### Safeguards built in (why each loop iteration can "skip")
- **All-zero pose** → PLC raised the trigger before writing `D2000`; wait, don't record.
- **Pose changed between two reads** → PLC mid-write/desync; skip this cycle.
- **NaN camera pose** → bad detection; retry frame.
- **Board reproj > `max_reproj_px`** → ambiguous/flipped board; reject (in `board_detect`).
- **Board Z outside 50–3000 mm** → not really in front of the camera; reject.

---

## How to run it

1. **Print/mount the ChArUco board.** Its geometry must match `config.yaml → charuco:`
   (`squares_x`, `squares_y`, `square_length`, `marker_length`, `dictionary`). Get this
   wrong and every pose is wrong.
2. **Set up the robot program** to step through ~16 varied poses, raising `M2000` at each
   and waiting for `M2001`. **Vary the orientation a lot** (rotate about ≥2 axes) — pure
   translation gives a degenerate, useless solve.
3. **Tune `calibration:` in config** if needed (`total_poses`, `min_charuco_corners`,
   `max_reproj_px`, `settle_sec`, `pose_stable_tol`).
4. Run:
   ```bash
   python calibration/aruco_calibate.py
   ```
   Watch the live window: it shows corner count, the two handshake bits, and what it's
   waiting on. Press `q` to abort.
5. After `total_poses` captures it solves, prints all 5 methods for comparison, and saves:
   - `calibration/hand_eye_result.npz` — R/t + **all raw per-pose matrices**.
   - `calibration/capture_log.csv` — raw KUKA pose, raw PLC words, reproj error per pose.

---

## How to know it worked (verification)

The script prints a **residual**: since the board is fixed in the world, its computed
position in the **base frame** must be the same point for every pose. The spread of those
points = solve error.

- **RMS residual < ~50 mm** → consistent, usable. (We've achieved ~29 mm.)
- **> 50 mm** → NOT reliable. Likely degenerate poses (vary A/B/C more) or a
  units/transform-direction mismatch.

If it's bad, **don't guess** — use the offline debugger
[`tools/cal_ressult_calib.py`](../../tools/cal_ressult_calib.py). It re-loads the CSV+NPZ
and brute-forces all 5 solver methods × 32 angle conventions/signs, plus a pair-consistency
diagnostic that flags individual bad poses to drop. See [tools.md](tools.md).

---

## Putting the result into production

`aruco_calibate.py` saves to `.npz`, but `main.py` reads the calibration from
**`config.yaml`** (`robot.hand_eye_rotation` / `robot.hand_eye_translation`). After a good
calibration, copy the PARK `R_cam2gripper` (3×3) and `t_cam2gripper` (meters) into those
config keys. That's the manual handoff step — there's no auto-write.

> **Units:** the calibration feeds robot translations in **mm**, so the printed/saved
> `t_cam2gripper` is in **mm**. `config.yaml → robot.hand_eye_translation` is in **meters**.
> Divide by 1000 when copying (e.g. `-44.9 mm → -0.044947`). `R_cam2gripper` is unitless,
> copy as-is.
