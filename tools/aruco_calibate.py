import os
import sys
import csv
import time
import struct

import cv2
import numpy as np
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from communication.realsense import DepthCamera
from communication.plc_comm import PLCCommunicator

from tools.geometry import rotation_matrix_from_abc, marker_positions_in_base, residual_stats
from tools.plc_decode import decode_pose
from tools.board_detect import build_charuco, detect_board_pose

# ---------------------------------------------------------------------------
# Config  (edit the robot-handshake registers to match your PLC program)
# ---------------------------------------------------------------------------
TRIGGER_DEVICE = "M2000"   # BIT  - PLC sets =1 when the KUKA has reached the pose
ACK_DEVICE = "M2001"       # BIT  - PC sets =1 after recording ("camera complete ok")
POSE_DEVICE = "D2000"      # WORD - start of 6 double-words: X Y Z A B C (32-bit each)
POSE_WORD_COUNT = 12       # 6 double-words * 2 words each
TOTAL_POINTS_NEEDED = 20
MIN_CHARUCO_CORNERS = 8     # min chessboard corners needed for a stable board pose
MAX_REPROJ_PX = 2.0        # reject a board pose whose reprojection error exceeds this (kills 180-deg flips)
SETTLE_SEC = 0.5           # wait this long after M2000=1 before recording (robot settle + stable pose words)
POSE_STABLE_TOL = 0.5      # max change (mm/deg) allowed between two consecutive pose reads before recording

WORD_SWAP = False           # set True if KUKA REALs come back high-word-first (fixes nan/0.0 decodes)
DEBUG_RAW_POSE = True       # print raw words + both decodings each capture; set False once verified
DEBUG_HANDSHAKE = True      # print M2000/M2001 + what the loop is waiting for on every change


def load_config():
    with open(os.path.join(PROJECT_ROOT, "config.yaml"), "r") as f:
        return yaml.safe_load(f)


def read_robot_pose(plc, verbose=DEBUG_RAW_POSE):
    """Read the robot pose from the PLC. Returns (pose_tuple_or_None, raw_words).

    pose_tuple is 6 floats (X Y Z A B C, mm/deg). raw_words is always returned so
    the caller can log it for offline diagnosis.
    """
    raw_words = plc.read_words(POSE_DEVICE, POSE_WORD_COUNT)
    if verbose:
        print("   raw words:", " ".join(f"{w & 0xFFFF:04X}" for w in raw_words))
        for label, sw in (("normal ", False), ("swapped", True)):
            try:
                vals = tuple(round(v, 2) for v in decode_pose(raw_words, sw))
                print(f"   decode {label}: {vals}")
            except Exception as e:
                print(f"   decode {label}: error {e}")
    try:
        return decode_pose(raw_words, WORD_SWAP), raw_words
    except struct.error:
        return None, raw_words


def read_stable_pose(plc):
    """Read the pose twice; return it only if it didn't move between reads.

    Guards against recording a pose while the PLC is mid-write or still holds the
    previous value (a desync that scrambles the hand-eye pose<->image pairing).
    Returns (pose_or_None, raw_words, stable: bool).
    """
    pose1, _ = read_robot_pose(plc)
    time.sleep(0.05)
    pose2, raw_words = read_robot_pose(plc, verbose=False)
    if pose1 is None or pose2 is None:
        return None, raw_words, False
    stable = max(abs(a - b) for a, b in zip(pose1, pose2)) <= POSE_STABLE_TOL
    return pose2, raw_words, stable


def _waiting_for(state, trigger, board_ok, n_corners):
    """Short status describing what the handshake loop is currently blocked on."""
    if state == "wait_release":
        return "waiting PLC release"
    # wait_trigger
    if not trigger:
        return "waiting trigger"
    if not board_ok:
        return f"bad board (saw {n_corners} corners, need {MIN_CHARUCO_CORNERS})"
    return "recording"


def main():
    config = load_config()
    plc_cfg = config["plc"]
    cam_cfg = config["camera"]
    ch_cfg = config.get("charuco", {})

    squares_x = int(ch_cfg.get("squares_x", 8))
    squares_y = int(ch_cfg.get("squares_y", 11))
    square_mm = float(ch_cfg.get("square_length", 0.0236)) * 1000.0
    marker_mm = float(ch_cfg.get("marker_length", 0.0153)) * 1000.0
    dict_name = ch_cfg.get("dictionary", "DICT_6X6_250")

    save_dir = os.path.join(PROJECT_ROOT, "output")
    os.makedirs(save_dir, exist_ok=True)

    board, charuco_detector = build_charuco(squares_x, squares_y, square_mm, marker_mm, dict_name)

    # PLC is mandatory: capture is driven by the robot trigger bit.
    plc = PLCCommunicator(plc_cfg["ip"], plc_cfg["port"])
    if not plc.connect():
        print("❌ PLC connection failed. PLC-trigger mode needs the PLC online. Aborting.")
        return

    camera = DepthCamera(cam_cfg["resolution_width"], cam_cfg["resolution_height"])
    camera_matrix, dist_coeffs = camera.get_color_intrinsics()
    print(f"✅ RealSense ready | ChArUco {squares_x}x{squares_y} "
          f"sq={square_mm:.1f}mm mk={marker_mm:.1f}mm | dict={dict_name}")
    print(f"--- Collecting {TOTAL_POINTS_NEEDED} poses. PLC raises {TRIGGER_DEVICE} when KUKA is parked; "
          f"PC pulses {ACK_DEVICE} when done. Press [q] to abort. ---")

    R_gripper2base, t_gripper2base = [], []
    R_target2cam, t_target2cam = [], []
    capture_log = []   # one row per recorded pose, raw inputs for offline diagnosis
    count = 0
    plc.write_bit(ACK_DEVICE, 0)  # start with the ack low

    # Interlocked 4-phase handshake (per calibrate.md):
    #   PLC sets M2000=1 (KUKA reached) -> PC records + sets M2001=1 (complete)
    #   -> PLC sees M2001=1, lowers M2000=0 -> PC sees that, lowers M2001=0 -> next cycle.
    # M2001 is HELD high until the PLC drops M2000, so the ack can't be missed.
    # States: wait_trigger -> wait_release
    state = "wait_trigger"
    last_hs = None         # last handshake status string printed (avoids flooding every frame)
    trigger_since = None   # time M2000 first went high this cycle (for SETTLE_SEC dwell)
    zero_warned = False    # throttle the all-zero-pose warning to once per occurrence

    try:
        while count < TOTAL_POINTS_NEEDED:
            ok, _, frame = camera.get_frame()
            if not ok or frame is None:
                continue

            success, R_t2c, t_t2c, debug, n_corners, reproj_px = detect_board_pose(
                frame, charuco_detector, board, camera_matrix, dist_coeffs,
                min_corners=MIN_CHARUCO_CORNERS, max_reproj_px=MAX_REPROJ_PX)

            # Read both handshake bits up front so we can show/log live status.
            trigger = plc.read_bit(TRIGGER_DEVICE)[0] == 1
            ack = plc.read_bit(ACK_DEVICE)[0] == 1
            waiting = _waiting_for(state, trigger, success, n_corners)

            cv2.putText(debug, f"Captured: {count}/{TOTAL_POINTS_NEEDED}  [{state}]",
                        (10, debug.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if DEBUG_HANDSHAKE:
                cv2.putText(debug, f"{TRIGGER_DEVICE}={int(trigger)}  {ACK_DEVICE}={int(ack)}",
                            (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(debug, f"waiting: {waiting}", (10, 135),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
                hs_key = (trigger, ack, state)     # ignore fluctuating marker count
                if hs_key != last_hs:              # only print when bit/state actually changes
                    print(f"   M2000={int(trigger)} M2001={int(ack)} | {waiting}")
                    last_hs = hs_key

            cv2.imshow("Hand-Eye Calibration (ArUco)", debug)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                print("❌ Aborted by user.")
                break

            if state == "wait_release":
                if not trigger:                    # PLC dropped M2000 after seeing our ack
                    plc.write_bit(ACK_DEVICE, 0)   # now release M2001 -> ready for next pose
                    state = "wait_trigger"
                continue

            # state == "wait_trigger": only act once the KUKA has reached the pose.
            if not trigger:
                trigger_since = None       # robot not parked yet; reset the settle timer
                zero_warned = False
                continue

            # Let the robot settle (and the pose words stabilise) before recording.
            if trigger_since is None:
                trigger_since = time.time()
            settle_left = SETTLE_SEC - (time.time() - trigger_since)
            if settle_left > 0:
                continue

            if not success:
                cv2.putText(debug, "Triggered - waiting for board...", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                continue

            pose, raw_words, stable = read_stable_pose(plc)
            if pose is None:
                print("⚠️ Failed to read robot pose from PLC — retrying this cycle.")
                continue
            if not stable:
                print("⚠️ Robot pose changed between reads — PLC mid-write/desync. Skipping this cycle.")
                continue
            X, Y, Z, A, B, C = pose

            # =================================================================
            if np.isnan(t_t2c).any() or np.isnan(R_t2c).any():
                print("⚠️ [Warning] Camera Pose contains NaN! Retrying this frame...")
                continue # ข้ามลูปนี้ไปสแกนภาพเฟรมใหม่ โดยไม่นับแต้ม ไม่ส่ง ACK
            # Reject empty pose words: PLC raised the trigger before writing D2000.
            # Recording (0,0,0) here corrupts the hand-eye solve, so wait instead.
            # =================================================================

            if X == 0.0 and Y == 0.0 and Z == 0.0:
                if not zero_warned:
                    print("⚠️ Robot pose reads all-zero — PLC hasn't written the pose yet. "
                          "Waiting (not recording, not acking).")
                    zero_warned = True
                continue
            zero_warned = False

            R_gripper2base.append(rotation_matrix_from_abc(A, B, C))
            t_gripper2base.append(np.array([[X], [Y], [Z]]))
            R_target2cam.append(R_t2c)
            t_target2cam.append(t_t2c)
            capture_log.append({
                "idx": count, "X": X, "Y": Y, "Z": Z, "A": A, "B": B, "C": C,
                "cam_x": float(t_t2c[0][0]), "cam_y": float(t_t2c[1][0]), "cam_z": float(t_t2c[2][0]),
                "reproj_px": reproj_px, "n_corners": n_corners,
                "raw_words": " ".join(f"{w & 0xFFFF:04X}" for w in raw_words),
            })
            count += 1

            plc.write_bit(ACK_DEVICE, 1)           # "camera complete ok" — held until PLC drops M2000
            state = "wait_release"
            trigger_since = None                   # re-arm settle timer for the next pose

            print(f" [{count}/{TOTAL_POINTS_NEEDED}]  reproj={reproj_px:.2f}px corners={n_corners}")
            print(f"    Cam   X:{t_t2c[0][0]:7.1f} Y:{t_t2c[1][0]:7.1f} Z:{t_t2c[2][0]:7.1f} mm")
            print(f"    Robot X:{X:7.1f} Y:{Y:7.1f} Z:{Z:7.1f} | A:{A:6.1f} B:{B:6.1f} C:{C:6.1f}")
    finally:
        plc.write_bit(ACK_DEVICE, 0)
        camera.release()
        cv2.destroyAllWindows()
        plc.disconnect()

    if count < TOTAL_POINTS_NEEDED:
        print(f"\nCollected only {count} poses (<{TOTAL_POINTS_NEEDED}). Calibration skipped.")
        return

    print("\n--- Computing Eye-in-Hand calibration ---")

    # Cross-method agreement: if these disagree a lot, the input poses are bad.
    methods = {
        "TSAI": cv2.CALIB_HAND_EYE_TSAI,
        "PARK": cv2.CALIB_HAND_EYE_PARK,
        "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
        "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
        "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }
    print("\nMethod comparison (t_cam2gripper, mm) — should agree if data is good:")
    solved = {}
    for name, m in methods.items():
        R_m, t_m = cv2.calibrateHandEye(
            R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, method=m)
        solved[name] = (R_m, t_m)
        print(f"  {name:11s} X:{t_m[0][0]:9.2f} Y:{t_m[1][0]:9.2f} Z:{t_m[2][0]:9.2f}  |t|={np.linalg.norm(t_m):8.2f}")

    R_cam2gripper, t_cam2gripper = solved["TSAI"]

    # Residual: the marker is fixed in the world, so its position computed in the
    # robot base frame must be the SAME point for every pose. Spread = solve error.
    pts_base = marker_positions_in_base(R_gripper2base, t_gripper2base, t_target2cam,
                                        R_cam2gripper, t_cam2gripper)
    mean_pt, rms_resid, max_resid = residual_stats(pts_base)

    print("\n=======================================================")
    print(" EYE-IN-HAND RESULT (TSAI, mm) ")
    print(f" X Offset : {t_cam2gripper[0][0]:.3f} mm")
    print(f" Y Offset : {t_cam2gripper[1][0]:.3f} mm")
    print(f" Z Offset : {t_cam2gripper[2][0]:.3f} mm")
    print("\nRotation Matrix (cam -> gripper):\n", R_cam2gripper)
    print("-------------------------------------------------------")
    print("RESIDUAL — marker position spread in base frame (lower = better):")
    print(f"  mean marker pos in base : {mean_pt[0]:.1f}, {mean_pt[1]:.1f}, {mean_pt[2]:.1f} mm")
    print(f"  RMS residual            : {rms_resid:.2f} mm")
    print(f"  max  residual           : {max_resid:.2f} mm")
    if max_resid > 50:
        print("  ⚠️  residual is large (>50mm): calibration is NOT reliable.")
        print("      Likely degenerate poses (vary A/B/C more, about >=2 axes),")
        print("      or a units/transform-direction mismatch in the robot pose.")
    else:
        print("  ✅ residual small: calibration looks consistent.")
    print("=======================================================")

    result_path = os.path.join(save_dir, "hand_eye_result.npz")
    np.savez(result_path, R_cam2gripper=R_cam2gripper, t_cam2gripper=t_cam2gripper,
             R_gripper2base=np.array(R_gripper2base), t_gripper2base=np.array(t_gripper2base),
             R_target2cam=np.array(R_target2cam), t_target2cam=np.array(t_target2cam))
    print(f"Saved: {result_path}  (includes raw per-pose data for offline diagnosis)")

    # Raw per-capture log: raw KUKA pose + words + reprojection error, so a bad
    # run can be replayed/audited offline (was missing before — couldn't debug).
    log_path = os.path.join(save_dir, "capture_log.csv")
    fields = ["idx", "X", "Y", "Z", "A", "B", "C",
              "cam_x", "cam_y", "cam_z", "reproj_px", "n_corners", "raw_words"]
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(capture_log)
    print(f"Saved: {log_path}  (raw KUKA pose, words, and reprojection error per capture)")


if __name__ == "__main__":
    main()
