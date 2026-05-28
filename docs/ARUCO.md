# ArUco Reference Frame

This project uses a single ArUco marker as the **world origin** for all 3D
measurements sent to the PLC. The robot consumes target positions in the
marker's local frame, so the marker effectively defines "robot coordinates".

---

## 1. Why ArUco?

- The RealSense camera natively reports points in its own optical frame.
- The robot needs positions relative to a fixed reference on the workspace.
- An ArUco marker (printed and fixed in the workspace) gives the system a
  cheap, reliable way to recover that reference every trigger.

Each scan:

```
camera frame -> ArUco frame -> PLC (robot)
```

---

## 2. Configuration

Edit the `aruco` section in [config.yaml](../config.yaml):

```yaml
aruco:
  enabled: true             # false = skip ArUco entirely (output stays in camera frame)
  marker_length: 0.055      # METERS, edge length of the printed marker
  dictionary: DICT_6X6_250  # any cv2.aruco.DICT_* name
```

`marker_length` must match the **physical printed size**. A wrong value
scales pose estimation linearly:

```
actual_length = config_length * (printed_Z / measured_Z)
```

Example: marker is 58 cm from the camera but `[ARUCO POSE]` prints
`Z = +0.50 m`, then real marker edge ≈ `0.055 * 0.50 / 0.58 ≈ 0.0474 m`.

---

## 3. Marker frame convention

Marker corners are defined as a square centered at the marker origin:

```
   ( -L/2, +L/2, 0 ) +--------+ ( +L/2, +L/2, 0 )
                     |        |
                     |   Z    |   Z points OUT of the page (toward camera)
                     |   ^    |   Y points UP
                     |        |   X points RIGHT
   ( -L/2, -L/2, 0 ) +--------+ ( +L/2, -L/2, 0 )
```

`(0, 0, 0)` = marker center.

All XYZ values sent to the PLC are expressed in this frame.

---

## 4. Runtime flow

Per trigger ([main.py](../main.py)):

1. Capture color + depth from RealSense.
2. Run template detection to find target pixels.
3. `PointCloudTransformer.extract_3d_data()` lifts pixels to 3D in the
   camera frame.
4. `ArucoReference.detect_pose()` finds the marker on the same frame
   and returns `(rvec, tvec)`.
   - If the marker is not visible, the cycle aborts with
     `ERR_NO_REFERENCE = 4` and zero targets.
5. Each target's `(x, y, z)` is converted to the marker frame via
   `ArucoReference.transform_from_transformer_frame()` and then scaled
   and written to the PLC.
6. The 3D viewer geometries are re-expressed in the marker frame so the
   marker sits at `(0, 0, 0)` — press `p` after a trigger to inspect.

---

## 5. Live overlay (2D camera window)

`ArucoReference.draw_overlay()` runs every frame on the live preview:

- Detected marker outline (cyan/yellow)
- 3D axes drawn on the marker (OpenCV convention: X-red, Y-green, Z-blue)
- Status text at the bottom of the window:
  - Green: `ArUco id=N  t=(x, y, z) m` when detected
  - Red:   `ArUco: NOT DETECTED` when missing

Note: the 2D overlay uses OpenCV's convention (Y points *down* in the
image), while the 3D viewer uses a flipped world where Y points *up*.
This is purely cosmetic — both represent the same physical directions.

---

## 6. Terminal output

Every successful trigger prints the marker pose in the **camera frame**:

```
[ARUCO POSE] id=7  pos=(+0.1234, -0.0456, +0.5800) m  rot=(roll=+1.20, pitch=-3.45, yaw=+88.10) deg (in camera frame)
```

Followed by each target's coordinates in the **marker frame**:

```
[PLC] slot 0 (PointA/PointA.1)
  float : X=+0.0521m  Y=-0.0033m  Z=+0.0120m  Conf= 87.50%
  sent  : X=  +521   Y=    -33   Z=   +120   Conf=  8750   (x10000 / x100)
```

---

## 7. Error code

| Code | Meaning                                  |
| ---- | ---------------------------------------- |
| 0    | OK                                       |
| 1    | Invalid program no.                      |
| 2    | No targets found                         |
| 3    | Camera error                             |
| 4    | ArUco reference marker not visible       |
| 99   | Internal error                           |

Error `4` only occurs when `aruco.enabled: true` and the marker is missing
from the trigger frame.

---

## 8. Files

| File                                                  | Role                                                                 |
| ----------------------------------------------------- | -------------------------------------------------------------------- |
| [core/aruco_reference.py](../core/aruco_reference.py) | Detection, pose estimation, coordinate transform, live overlay       |
| [communication/realsense.py](../communication/realsense.py) | `get_color_intrinsics()` provides camera matrix + distortion coeffs |
| [core/transformer.py](../core/transformer.py)         | `re_express_in_marker_frame()` moves the 3D viewer into marker frame |
| [main.py](../main.py)                                 | Wires everything together in the trigger loop                        |
| [config.yaml](../config.yaml)                         | Marker size, dictionary, enable flag                                 |

---

## 9. Troubleshooting

| Symptom                                              | Likely cause / fix                                                                |
| ---------------------------------------------------- | --------------------------------------------------------------------------------- |
| `ArUco: NOT DETECTED` (live overlay)                 | Marker out of view, occluded, blurry, or poor lighting                            |
| Reported Z is wrong by a constant factor             | `marker_length` in config doesn't match the printed marker                        |
| Marker detected but axes jitter heavily              | Marker too small / too far / motion blur; print larger marker, slow scene motion  |
| Target XYZ unrealistic after enabling ArUco          | Verify marker pose first (`[ARUCO POSE]` line); if that's wrong, fix size first   |
| Want to bypass ArUco temporarily                     | Set `aruco.enabled: false` in [config.yaml](../config.yaml) — outputs stay in camera frame |
