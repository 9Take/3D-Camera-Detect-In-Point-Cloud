# 2. Core — The Vision Brain

> 🌐 Language: **English** | [ไทย](core.th.md)

The `core/` package is where pixels become 3D poses. Two files:

- [`core/detector.py`](../../core/detector.py) — **2D**: find taught features in the color image.
- [`core/transformer.py`](../../core/transformer.py) — **2D → 3D**: turn a found pixel into a 6-DOF pose.

---

## 2.1 `detector.py` — `ObjectDetector`

### Job
Given a color frame, find every taught template in it and return, for each hit:
the **target pixel** `(u, v)`, the **template name**, a **confidence** %, and the
**bounding polygon** (homography corners) for drawing.

### How it works
1. **At startup** — `_load_templates(template_dir)` walks the program folder. For every
   `*_template.png` inside a `Point*/` subfolder it:
   - precomputes SIFT keypoints + descriptors **once** (so per-frame matching is fast),
   - reads the **target offset** (the exact "click point" inside the template) from the
     matching `*_meta.json` (`offset_x`, `offset_y`). Falls back to `*_offset.txt`, then to
     the template center.
   - stores `{img, offset, kp, des, point}` keyed by target name. `point` is the parent
     `PointA`/`PointB` folder — used later for "best per point".
2. **Per frame** — `detect(color_frame, w, h)`:
   - SIFT on the grayscale frame.
   - For each template: **FLANN k-NN (k=2)** match + **Lowe ratio test (0.7)** to keep
     good matches. Need **> 12** good matches to continue.
   - **RANSAC homography** template → frame.
   - **Confidence** = `min(100, inliers / 30 * 100)`.
   - Project the template's offset point through the homography → the target pixel in the
     live frame. Project the 4 corners → the bounding polygon.
   - Keeps the hit only if the target pixel is inside the frame.

### Return value (order matters — callers unpack positionally)
```python
detected_pixels, detected_names, detected_confidences, detected_homographies, display_frame
```

### `build_sub_window_grid(...)`
Builds a cropped "thumbnail grid" of all detections for visual review (one tile per hit
with its outline + confidence). `main.py` has its own richer version
(`_build_trigger_result_grid`) that also highlights the best per point, so this method is
secondary.

### Tuning knobs (hard-coded in `detect()`)
| What | Value | Where |
|------|-------|-------|
| Lowe ratio | `0.7` | line ~71 |
| Min good matches | `> 12` | line ~73 |
| RANSAC reproj threshold | `5.0` px | `findHomography` |
| Confidence scaling | `inliers / 30` | line ~81 |

If detections are too sparse, loosen the ratio (e.g. 0.75) or lower the min-matches. If
you get false matches, tighten them.

---

## 2.2 `transformer.py` — `PointCloudTransformer`

### Job
Given the list of target pixels + names, read depth, back-project to 3D, estimate the
local surface orientation, and return a 6-DOF pose **per target** in the camera/point-cloud
frame. `main.py` then transforms these into the robot base frame.

### `extract_3d_data(target_pixels, target_names, show_3d=True)` step by step
1. Grab a fresh aligned RGB-D frame from the camera (`get_raw_frame`).
2. Build an Open3D point cloud from the RGB-D image using the RealSense intrinsics, then
   `transform([...flip Y and Z...])` so the world is right-handed (up = +Y, forward = −Z).
3. Estimate per-point **normals** (`KDTreeSearchParamHybrid(radius=0.01, max_nn=30)`).
4. For each target pixel `(u, v)`:
   - **Depth-zero recovery (the copper-shine fix):** if depth at the pixel is 0, expand a
     search radius from 2→7 px and average the non-zero neighbors. If a 15×15 window is
     still all zero, **skip** this target (reported upstream as `no_targets`).
   - **Back-project** with the intrinsics:
     ```
     Z = depth_raw * depth_scale
     X = (u - cx) * Z / fx
     Y = (v - cy) * Z / fy
     ```
   - Find the nearest point-cloud point and take its **normal as the local Z axis**.
     Build X and Y axes perpendicular to it (with a stable seed to avoid degeneracy) →
     a 3×3 rotation matrix.
   - Convert to Euler `(roll, pitch, yaw)`.
   - Store: `[X, -Y, -Z, roll, pitch, yaw, rotation_matrix]`.
     **Note index 6 = the full 3×3 matrix** — `main.py` uses that (not the Euler angles)
     to transform orientation into the base frame. The Euler values are for logging.
5. Caches the cloud + per-target spheres + axes in `self._last_geometries` so the operator
   can open the 3D viewer later (`p` key) without re-running detection.

### Other methods
- `show_collected_3d(...)` — opens the Open3D viewer on the cached geometries (blocks until
  the window is closed). Wired to the `p` key in `main.py`.
- `re_express_in_marker_frame(rvec, tvec)` — re-centers the whole cached scene on an ArUco
  marker (debug aid for verifying the transform visually). Not used in the runtime path.

### Frames — the part that confuses everyone
There are three frames in play. Keep them straight:

| Frame | +X | +Y | +Z | Used by |
|-------|----|----|----|---------|
| OpenCV camera | right | down | forward | back-projection, hand-eye |
| Point-cloud (Open3D, flipped) | right | up | backward | `extract_3d_data` output |
| Robot base (KUKA) | — | — | — | what the PLC wants |

`extract_3d_data` returns the **point-cloud** frame (Y/Z flipped). `main.py`'s
`encode_target_pose` flips Y and Z back to **camera** frame, then applies the
`H_cam2base` transform to reach the **base** frame. See [main-loop.md](main-loop.md).
