import cv2
import time
import yaml
import json
import os
import argparse
import numpy as np

from communication.realsense import DepthCamera
from communication.plc_comm import PLCCommunicator
from core.detector import ObjectDetector
from core.transformer import PointCloudTransformer


# Error codes written to plc.error_code_device
ERR_OK = 0
ERR_INVALID_PROGRAM = 1
ERR_NO_TARGETS = 2
ERR_CAMERA = 3
ERR_INTERNAL = 99


def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def load_detectors(programs_cfg):
    """Pre-load one ObjectDetector per program. Returns {program_no: (name, detector)}."""
    detectors = {}
    for pno, pdef in programs_cfg.items():
        name = pdef['name']
        tdir = pdef['template_dir']
        print(f"\n[INIT] Loading program {pno} ({name}) from {tdir}")
        detectors[int(pno)] = (name, ObjectDetector(tdir))
    return detectors


def best_per_point(detector, detected_names, confidences):
    """Group detections by their parent point (PointA, PointB, ...) and keep the best per point."""
    best = {}
    for idx, (name, conf) in enumerate(zip(detected_names, confidences)):
        meta = detector.templates_by_target.get(name)
        if meta is None:
            continue
        point = meta['point']
        if point not in best or conf > best[point]['conf']:
            best[point] = {'idx': idx, 'conf': conf, 'name': name}
    return best


def set_status(plc, cfg, ready=None, busy=None, complete=None, error=None):
    if ready    is not None: plc.write_bit(cfg['status_ready_device'],    ready)
    if busy     is not None: plc.write_bit(cfg['status_busy_device'],     busy)
    if complete is not None: plc.write_bit(cfg['status_complete_device'], complete)
    if error    is not None: plc.write_bit(cfg['status_error_device'],    error)


def main():
    parser = argparse.ArgumentParser(description="Heat Exchanger Vision System")
    parser.add_argument('--debug', action='store_true', help='Show 2D detection + 3D point cloud')
    parser.add_argument('--show2d', action='store_true', help='Show 2D detection only')
    args = parser.parse_args()

    config = load_config()
    plc_cfg = config['plc']
    os.makedirs(config['paths']['save_dir'], exist_ok=True)

    print("\n[INIT] Initializing Systems...")
    cam = DepthCamera(config['camera']['resolution_width'], config['camera']['resolution_height'])
    detectors = load_detectors(config['programs'])
    transformer = PointCloudTransformer(cam, config['camera']['resolution_width'],
                                        config['camera']['resolution_height'],
                                        config['paths']['save_dir'])

    plc = PLCCommunicator(plc_cfg['ip'], plc_cfg['port'])
    plc.connect()
    plc.start_heartbeat(plc_cfg['heartbeat_device'], plc_cfg.get('heartbeat_interval_sec', 1.0))

    # Initial PLC state: idle and ready
    plc.write_word(plc_cfg['error_code_device'], ERR_OK)
    set_status(plc, plc_cfg, ready=1, busy=0, complete=0, error=0)

    pos_mul  = plc_cfg.get('position_multiplier', 10000)
    conf_mul = plc_cfg.get('confidence_multiplier', 100)
    words_per_slot = plc_cfg.get('words_per_slot', 4)
    max_points = plc_cfg.get('max_points', 5)

    # Last program seen — used for debug preview before the first trigger arrives
    preview_pno = sorted(detectors.keys())[0]

    print("\n[SYSTEM READY] Waiting for PLC trigger...")

    try:
        while True:
            ret, depth_raw, color_raw = cam.get_raw_frame()
            if not ret: continue
            color_frame = np.asanyarray(color_raw.get_data())

            # --- Debug preview (does not write PLC) ----------------------------
            if args.debug or args.show2d:
                p_name, p_det = detectors[preview_pno]
                detected_pixels, detected_names, confidences, detected_homographies, _ = p_det.detect(
                    color_frame, config['camera']['resolution_width'], config['camera']['resolution_height']
                )
                best = best_per_point(p_det, detected_names, confidences)

                main_display = color_frame.copy()
                cv2.putText(main_display, f"Program {preview_pno}: {p_name}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                for g_idx, point in enumerate(sorted(best)):
                    entry = best[point]
                    pixel = detected_pixels[entry['idx']]
                    cv2.circle(main_display, pixel, 8, (0, 255, 0), -1)
                    cv2.putText(main_display, f"BEST {point}: {entry['name']} ({entry['conf']:.1f}%)",
                                (pixel[0] - 40, pixel[1] - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                window_title = "Vision System - Main [1-9=Trigger Program | ESC=Quit]" if args.debug else "2D Detection"
                cv2.imshow(window_title, main_display)
                grid_img = p_det.build_sub_window_grid(color_frame, detected_pixels, detected_names,
                                                      confidences, detected_homographies)
                cv2.imshow("Detected Sub-Windows Grid", grid_img)
                cv2.setWindowProperty("Detected Sub-Windows Grid", cv2.WND_PROP_TOPMOST, 1)
                key = cv2.waitKey(1) & 0xFF
            else:
                key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

            # --- Trigger check -------------------------------------------------
            trigger_bit = plc.read_bit(plc_cfg['trigger_device'])[0]
            manual_program = None
            if args.debug and ord('1') <= key <= ord('9'):
                manual_program = key - ord('0')
            manual_trigger = manual_program is not None
            if not (trigger_bit == 1 or manual_trigger):
                continue

            # ---- Cycle start ----
            set_status(plc, plc_cfg, ready=0, busy=1, complete=0, error=0)
            plc.write_word(plc_cfg['error_code_device'], ERR_OK)

            program_no = manual_program if manual_trigger else plc.read_word(plc_cfg['program_no_device'])
            print(f"\n[TRIGGER] Program No. = {program_no}{'  (manual)' if manual_trigger else ''}")

            if program_no not in detectors:
                print(f"[ERROR] Unknown program {program_no}")
                plc.write_word(plc_cfg['error_code_device'], ERR_INVALID_PROGRAM)
                set_status(plc, plc_cfg, ready=1, busy=0, complete=0, error=1)
                _wait_trigger_low(plc, plc_cfg)
                continue

            preview_pno = program_no
            prog_name, detector = detectors[program_no]
            print(f"[RUN] Using program '{prog_name}'")

            # Re-capture frame for the actual scan
            ret, _, color_raw = cam.get_raw_frame()
            if not ret:
                plc.write_word(plc_cfg['error_code_device'], ERR_CAMERA)
                set_status(plc, plc_cfg, ready=1, busy=0, complete=0, error=1)
                _wait_trigger_low(plc, plc_cfg)
                continue
            scan_frame = np.asanyarray(color_raw.get_data())

            detected_pixels, detected_names, confidences, _, _ = detector.detect(
                scan_frame, config['camera']['resolution_width'], config['camera']['resolution_height']
            )
            best = best_per_point(detector, detected_names, confidences)

            if not best:
                print("[WARNING] No targets found.")
                plc.write_word(plc_cfg['amount_device'], 0)
                plc.write_word(plc_cfg['error_code_device'], ERR_NO_TARGETS)
                set_status(plc, plc_cfg, busy=0, complete=1, error=1)
                _wait_trigger_low(plc, plc_cfg)
                set_status(plc, plc_cfg, ready=1, complete=0, error=0)
                continue

            filtered_pixels, filtered_names, filtered_confs, filtered_points = [], [], [], []
            for point in sorted(best):
                entry = best[point]
                filtered_pixels.append(detected_pixels[entry['idx']])
                filtered_names.append(entry['name'])
                filtered_confs.append(entry['conf'])
                filtered_points.append(point)
                print(f"[FILTER] {point} -> {entry['name']} (Conf: {entry['conf']:.2f}%)")

            # show_3d=False so PLC writes & prints happen immediately; PLY still saved in debug
            extracted_6dof = transformer.extract_3d_data(
                filtered_pixels, filtered_names,
                show_3d=False, save_ply=args.debug
            )

            if not extracted_6dof:
                plc.write_word(plc_cfg['amount_device'], 0)
                plc.write_word(plc_cfg['error_code_device'], ERR_NO_TARGETS)
                set_status(plc, plc_cfg, busy=0, complete=1, error=1)
                _wait_trigger_low(plc, plc_cfg)
                set_status(plc, plc_cfg, ready=1, complete=0, error=0)
                continue

            # Write results: amount + per-slot (X, Y, Z, Conf)
            num_targets = min(len(extracted_6dof), max_points)
            plc.write_word(plc_cfg['amount_device'], num_targets)

            memory_state = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "program_no": program_no,
                "program_name": prog_name,
                "targets": {}
            }

            slot_idx = 0
            for name, point, conf in zip(filtered_names, filtered_points, filtered_confs):
                if name not in extracted_6dof or slot_idx >= num_targets:
                    continue
                x, y, z = extracted_6dof[name][0:3]
                xi = int(round(x * pos_mul))
                yi = int(round(y * pos_mul))
                zi = int(round(z * pos_mul))
                ci = int(round(conf * conf_mul))
                plc.write_slot(
                    plc_cfg['slot_base_device'], slot_idx, words_per_slot,
                    [xi, yi, zi, ci],
                )
                print(f"[PLC] slot {slot_idx} ({point}/{name})")
                print(f"  float : X={x:+.4f}m  Y={y:+.4f}m  Z={z:+.4f}m  Conf={conf:6.2f}%")
                print(f"  sent  : X={xi:+6d}   Y={yi:+6d}   Z={zi:+6d}   Conf={ci:6d}"
                      f"   (x{pos_mul} / x{conf_mul})")
                memory_state["targets"][point] = {
                    "template": name,
                    "X": round(x, 4), "Y": round(y, 4), "Z": round(z, 4),
                    "Confidence": round(conf, 2),
                }
                slot_idx += 1

            with open(config['paths']['position_mem'], "w") as f:
                json.dump(memory_state, f, indent=4)

            print(f"[PLC] Wrote {num_targets} slot(s).")
            set_status(plc, plc_cfg, busy=0, complete=1, error=0)
            _wait_trigger_low(plc, plc_cfg)
            set_status(plc, plc_cfg, ready=1, complete=0)
            print("[CYCLE] Done. Ready for next trigger.\n")

            if args.debug:
                print("[DEBUG] Showing 3D viewer — close window to continue.")
                transformer.show_collected_3d()

    except KeyboardInterrupt:
        print("\n[INFO] Exiting program...")
    finally:
        set_status(plc, plc_cfg, ready=0, busy=0, complete=0, error=1)
        plc.disconnect()
        cam.release()
        cv2.destroyAllWindows()


def _wait_trigger_low(plc, plc_cfg, timeout_sec=10.0, poll_sec=0.05):
    """Block until the PLC clears its trigger bit (handshake) or timeout."""
    start = time.time()
    while time.time() - start < timeout_sec:
        if plc.read_bit(plc_cfg['trigger_device'])[0] == 0:
            return True
        time.sleep(poll_sec)
    print("[WARN] Timed out waiting for trigger to clear.")
    return False


if __name__ == '__main__':
    main()
