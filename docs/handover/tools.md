# 5. Tools — Helper Scripts

> 🌐 Language: **English** | [ไทย](tools.th.md)

The `tools/` folder holds standalone helpers. Two of them (`plc_decode`, `geometry`,
`board_detect`) are **pure libraries** imported by the runtime/calibration; the other two
are **scripts you run by hand**.

---

## 5.1 `create_template.py` — teach a new template (RUN BY HAND)

This is how you add a new point for the detector to find. It's an interactive
point-and-click tool over the live camera feed.

```bash
python tools/create_template.py -p ProgramA --point PointA -v 1
# or just `python tools/create_template.py` and answer the prompts
```

Arguments:
- `-p/--program` — program number (`1`) or name (`ProgramA`). Resolved against `config.yaml`.
- `--point` — point name (`PointA`; bare `A` becomes `PointA`).
- `-v/--variant` — variant tag (`1`, `front`, `side`…). The saved name is `PointA.1`.
- `--debug` — also save `.ply` files and show the 3D alignment.

Workflow on screen:
1. **Live view** → press **SPACEBAR** to freeze a frame.
2. **Annotate**: **left-click** to draw the bounding polygon around the feature;
   **right-click** to mark the exact target point (the click point). `s` = save,
   `r` = retake, `c` = clear.
3. **Tracking**: it live-matches your new template so you can confirm it locks on.
   Press **`q`** to compute the 3D data and save the meta JSON, or **ESC** to exit.

It writes into `data/templates/<Program>/<Point>/`:
- `PointA.1_template.png` — the cropped grayscale patch.
- `PointA.1_meta.json` — `offset_x/offset_y` (the click point inside the patch, **this is
  what the runtime detector reads**), plus the captured 3D position/orientation for record.

> The runtime detector only needs `_template.png` + the `offset_x/offset_y` in
> `_meta.json`. The 3D fields in the meta are informational.

---

## 5.2 `cal_ressult_calib.py` — offline calibration debugger (RUN BY HAND)

When a hand-eye calibration gives a bad residual, run this **without the robot** to find
out why. It reads `calibration/capture_log.csv` + `calibration/hand_eye_result.npz` and:

1. **Pair diagnostic** — for each consecutive pose pair, compares the gripper's relative
   rotation angle to the camera's. They must match (`AX=XB`). A mismatch > 8° flags a
   **flipped or desynced** pose and names the worst offender to drop/recapture.
2. **Brute-force** — tries all 5 hand-eye methods × 4 matrix orders × 8 sign combinations
   (32 conventions) and reports the combination with the lowest residual.

```bash
python tools/cal_ressult_calib.py
```

Use it to answer: "is my data bad, or is it the solver/convention?" (The pair diagnostic is
blind to direction/transpose errors and solver fragility — a green diagnostic + a failing
solve points at the **solver or translation**, not the rotation pairing.)

> Note: this script's header comment references a `cal.py` brute-force file; the live
> brute-force logic now lives here in `tools/cal_ressult_calib.py`.

---

## 5.3 `plc_decode.py` — int32 ↔ PLC words (LIBRARY, pure logic)

The single source of truth for how a robot pose / result is packed into PLC words. No PLC
connection needed, so it's unit-testable with hand-built word lists.

| Function | Does |
|----------|------|
| `decode_pose(words, swap=False)` | consecutive words → tuple of floats. Each value = 2 words (low, high), int32, ÷1000, sign-extended. |
| `encode_pose(values, swap=False)` | mm/deg floats → flat int16 word list (inverse of decode). |
| `int32_to_words(value, swap=False)` | one int32 → `[low, high]` as signed int16. |

`POSE_SCALE = 1000.0` — the PLC sends/receives pose values scaled ×1000 (mm→µm, deg→mdeg).

**`swap`** = word order. `swap=True` means high-word-first. This maps to
`config.yaml → plc.pose_word_swap`. If decoded poses look garbage/huge, this flag is the
first suspect.

---

## 5.4 `geometry.py` — KUKA angle maths (LIBRARY, pure numpy)

| Function | Does |
|----------|------|
| `rotation_matrix_from_abc(A, B, C)` | KUKA Euler (deg) → rotation matrix. KUKA ABC is **intrinsic Z-Y-X**: `Rz(A)@Ry(B)@Rx(C)`, which scipy's `"ZYX"` matches. |
| `marker_positions_in_base(...)` | maps the marker origin into the base frame for each pose (for the residual check). |
| `residual_stats(pts_base)` | `(mean_point, rms_mm, max_mm)` spread of those points = solve error. |

No OpenCV, no hardware — safe to import and test anywhere.

---

## 5.5 `board_detect.py` — ChArUco board pose (LIBRARY, needs OpenCV)

Used by the calibration to find the board and recover its pose.

| Function | Does |
|----------|------|
| `build_charuco(sx, sy, sq_mm, mk_mm, dict)` | builds the `(board, detector)` used everywhere. |
| `detect_board_pose(frame, ...)` | detects the board, solves its pose, returns `(success, R_target2cam, t_target2cam, debug_frame, n_corners, reproj_px)`. |

Important details baked in:
- **Bypasses the broken `matchImagePoints()`** (OpenCV 4.7.0) — builds obj↔img pairs from
  `getChessboardCorners()[ids]`. See [calibration.md](calibration.md).
- Uses the **planar IPPE solver**, which returns *both* mirror solutions for a flat board;
  keeps the one with lower reprojection error → defeats the ~180° flip.
- Rejects fits with reproj > `max_reproj_px` or board Z outside 50–3000 mm.
