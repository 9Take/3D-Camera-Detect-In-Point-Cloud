import os
import sys
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

# ---------------------------------------------------------------------------
# Config  (edit the robot-handshake registers to match your PLC program)
# ---------------------------------------------------------------------------
TRIGGER_DEVICE = "M2000"   # BIT  - PLC sets =1 when the KUKA has reached the pose
ACK_DEVICE = "M2001"       # BIT  - PC sets =1 after recording ("camera complete ok")
POSE_DEVICE = "D2000"      # WORD - start of 6 double-words: X Y Z A B C (32-bit each)
POSE_WORD_COUNT = 12       # 6 double-words * 2 words each
TOTAL_POINTS_NEEDED = 20
MIN_CHARUCO_CORNERS = 8     # min chessboard corners needed for a stable board pose
SETTLE_SEC = 0.5           # wait this long after M2000=1 before recording (robot settle + stable pose words)

WORD_SWAP = False           # set True if KUKA REALs come back high-word-first (fixes nan/0.0 decodes)
DEBUG_RAW_POSE = True       # print raw words + both decodings each capture; set False once verified
DEBUG_HANDSHAKE = True      # print M2000/M2001 + what the loop is waiting for on every change


def load_config():
    with open(os.path.join(PROJECT_ROOT, "config.yaml"), "r") as f:
        return yaml.safe_load(f)


def rotation_matrix_from_abc(A, B, C):
    """KUKA Euler angles (deg) -> rotation matrix.  R = Rz(A) @ Ry(B) @ Rx(C)."""
    a, b, c = np.radians(A), np.radians(B), np.radians(C)
    R_z = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    R_y = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    R_x = np.array([[1, 0, 0], [0, np.cos(c), -np.sin(c)], [0, np.sin(c), np.cos(c)]])
    return R_z @ R_y @ R_x


def detect_board_pose(frame, charuco_detector, board, camera_matrix, dist_coeffs):
    """Detect the ChArUco board. Returns (success, R_target2cam, t_target2cam, debug_frame, n_corners)."""
    debug = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ch_corners, ch_ids, _, _ = charuco_detector.detectBoard(gray)

    n = 0 if ch_ids is None else len(ch_ids)
    if n < MIN_CHARUCO_CORNERS:
        cv2.putText(debug, f"ChArUco corners: {n} (need {MIN_CHARUCO_CORNERS})", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return False, None, None, debug, n

    cv2.aruco.drawDetectedCornersCharuco(debug, ch_corners, ch_ids)
    obj_pts, img_pts = board.matchImagePoints(ch_corners, ch_ids)
    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, camera_matrix, dist_coeffs)
    if not ok:
        return False, None, None, debug, n

    cv2.drawFrameAxes(debug, camera_matrix, dist_coeffs, rvec, tvec, board.getSquareLength() * 2, 2)
    cv2.putText(debug, f"X:{tvec[0][0]:.1f} Y:{tvec[1][0]:.1f} Z:{tvec[2][0]:.1f} mm",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    R_target2cam, _ = cv2.Rodrigues(rvec)
    return True, R_target2cam, tvec, debug, n


POSE_SCALE = 1000.0   # PLC sends pose as int32 scaled x1000 (mm->um, deg->mdeg)


def _decode_floats(words, swap):
    """Decode consecutive PLC words into scaled int32 values (/POSE_SCALE).
    swap=True => high-word-first."""
    out = []
    for i in range(0, len(words) - 1, 2):
        lo, hi = words[i] & 0xFFFF, words[i + 1] & 0xFFFF
        if swap:
            lo, hi = hi, lo
        val = (hi << 16) | lo
        if val >= 0x80000000:          # sign-extend negative int32
            val -= 0x100000000
        out.append(val / POSE_SCALE)
    return tuple(out)


def read_robot_pose(plc):
    """Read 6 floats (X Y Z A B C, mm/deg) from the PLC. Returns tuple or None on failure."""
    raw_words = plc.read_words(POSE_DEVICE, POSE_WORD_COUNT)
    if DEBUG_RAW_POSE:
        print("   raw words:", " ".join(f"{w & 0xFFFF:04X}" for w in raw_words))
        for label, sw in (("normal ", False), ("swapped", True)):
            try:
                vals = tuple(round(v, 2) for v in _decode_floats(raw_words, sw))
                print(f"   decode {label}: {vals}")
            except Exception as e:
                print(f"   decode {label}: error {e}")
    try:
        return _decode_floats(raw_words, WORD_SWAP)
    except struct.error:
        return None


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

    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_mm, marker_mm, dictionary)
    charuco_detector = cv2.aruco.CharucoDetector(board)

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

            success, R_t2c, t_t2c, debug, n_corners = detect_board_pose(
                frame, charuco_detector, board, camera_matrix, dist_coeffs)

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

            pose = read_robot_pose(plc)
            if pose is None:
                print("⚠️ Failed to read robot pose from PLC — retrying this cycle.")
                continue
            X, Y, Z, A, B, C = pose

            # Reject empty pose words: PLC raised the trigger before writing D2000.
            # Recording (0,0,0) here corrupts the hand-eye solve, so wait instead.
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
            count += 1

            plc.write_bit(ACK_DEVICE, 1)           # "camera complete ok" — held until PLC drops M2000
            state = "wait_release"
            trigger_since = None                   # re-arm settle timer for the next pose

            print(f"📸 [{count}/{TOTAL_POINTS_NEEDED}]")
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

    print("\n--- Computing Eye-in-Hand calibration (Tsai) ---")
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base, R_target2cam, t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI)

    print("\n=======================================================")
    print("🎉 EYE-IN-HAND RESULT (mm) 🎉")
    print(f" X Offset : {t_cam2gripper[0][0]:.3f} mm")
    print(f" Y Offset : {t_cam2gripper[1][0]:.3f} mm")
    print(f" Z Offset : {t_cam2gripper[2][0]:.3f} mm")
    print("\nRotation Matrix (cam -> gripper):\n", R_cam2gripper)
    print("=======================================================")

    result_path = os.path.join(save_dir, "hand_eye_result.npz")
    np.savez(result_path, R_cam2gripper=R_cam2gripper, t_cam2gripper=t_cam2gripper)
    print(f"Saved: {result_path}")


if __name__ == "__main__":
    main()
