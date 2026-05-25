import cv2
import sys
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


# Status bits are written together in one packet to cut PLC load (4 packets -> 1).
# Assumes status devices are consecutive in order: ready, error, busy, complete
# (e.g. M1000, M1001, M1002, M1003). Cache holds last-written values so partial
# updates don't need an extra read.
_last_status = {'ready': 0, 'error': 0, 'busy': 0, 'complete': 0}

def set_status(plc, cfg, ready=None, busy=None, complete=None, error=None):
    if ready    is not None: _last_status['ready']    = int(bool(ready))
    if busy     is not None: _last_status['busy']     = int(bool(busy))
    if complete is not None: _last_status['complete'] = int(bool(complete))
    if error    is not None: _last_status['error']    = int(bool(error))
    plc.write_bits(cfg['status_ready_device'],
                   [_last_status['ready'], _last_status['error'],
                    _last_status['busy'],  _last_status['complete']])


def main():
    parser = argparse.ArgumentParser(description="Heat Exchanger Vision System")
    parser.add_argument('--debug', action='store_true', help='Also open the 3D point-cloud viewer after each trigger')
    args = parser.parse_args()

    config = load_config()
    plc_cfg = config['plc']
    os.makedirs(config['paths']['save_dir'], exist_ok=True)

    print("\n[INIT] Initializing Systems...")
    cam = DepthCamera(config['camera']['resolution_width'], config['camera']['resolution_height'])
    detectors = load_detectors(config['programs'])
    transformer = PointCloudTransformer(cam, config['camera']['resolution_width'],
                                        config['camera']['resolution_height'])

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

    # Sticky program number: None until the PLC sends a valid one (or operator
    # picks one in --debug). Stale PLC reads (0 / unknown) leave it untouched.
    current_program_no = None

    # Debug-only PLC-test sub-mode: 'b' toggles; while active, 1-9 writes to
    # program_no_test_device and 't' pulses trigger_test_device.
    plc_test_mode = False

    print("\n[SYSTEM READY] Waiting for PLC trigger...")

    # Throttle PLC polling independent of camera FPS to limit packet rate.
    plc_poll_interval = plc_cfg.get('poll_interval_sec', 0.1)  # 10 Hz default
    last_plc_poll = 0.0
    trigger_bit = 0

    try:
        while True:
            ret, depth_raw, color_raw = cam.get_raw_frame()
            if not ret: continue
            color_frame = np.asanyarray(color_raw.get_data())

            # --- (1) Poll PLC program no.; sticky if invalid -------------------
            # In debug mode without PLC-test, keyboard owns current_program_no
            # (otherwise the PLC poll would overwrite manual 1-9 presses each frame).
            poll_plc_now = (time.time() - last_plc_poll) >= plc_poll_interval
            if poll_plc_now and (not args.debug or plc_test_mode):
                plc_pno = plc.read_word(plc_cfg['program_no_device'])
                if plc_pno in detectors:
                    current_program_no = plc_pno

            # --- (2) Live preview: 2D detection + bounding box only -----------
            main_display = color_frame.copy()
            if current_program_no is not None:
                p_name, p_det = detectors[current_program_no]
                detected_pixels, detected_names, confidences, detected_homographies, _ = p_det.detect(
                    color_frame, config['camera']['resolution_width'], config['camera']['resolution_height']
                )
                best = best_per_point(p_det, detected_names, confidences)

                cv2.putText(main_display,
                            f"Program {current_program_no}: {p_name}",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                for point in sorted(best):
                    entry = best[point]
                    pixel = detected_pixels[entry['idx']]
                    homo  = detected_homographies[entry['idx']]
                    cv2.polylines(main_display, [np.int32(homo)], True, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.circle(main_display, pixel, 6, (0, 0, 255), -1)
                    cv2.putText(main_display, f"{point}: {entry['name']} ({entry['conf']:.1f}%)",
                                (pixel[0] - 40, pixel[1] - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            else:
                cv2.putText(main_display, "WAITING for Program No. from PLC", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

            if args.debug:
                if plc_test_mode:
                    hint = "[PLC TEST: 1-9=write D1500 | t=pulse M1500 | b=exit test | p=3D | ESC/q=Quit]"
                else:
                    hint = "[t=Trigger | 1-9=Set Program | b=PLC test | p=3D view | ESC/q=Quit]"
            else:
                hint = "[p=3D view | ESC/q=Quit]"
            cv2.putText(main_display, hint, (10, color_frame.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            if args.debug and plc_test_mode:
                cv2.putText(main_display, "PLC TEST MODE",
                            (color_frame.shape[1] - 230, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            cv2.imshow("Vision System - Live", main_display)

            prog_label = (f"P{current_program_no}:{detectors[current_program_no][0]}"
                          if current_program_no is not None else "P?:waiting")
            mode_label = "PLC-TEST " if (args.debug and plc_test_mode) else ""
            sys.stdout.write(
                f"\r[HB {plc.heartbeat_counter:5d}] {mode_label}{prog_label}  "
                f"waiting for trigger... "
            )
            sys.stdout.flush()

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

            # 'p' opens the 3D viewer for the most recent trigger's geometries
            if key == ord('p'):
                if getattr(transformer, "_last_geometries", None):
                    print("[VIEW] Showing 3D point cloud — close window to continue.")
                    transformer.show_collected_3d()
                else:
                    print("[VIEW] No trigger has run yet; nothing to show.")

            # --- Debug-only keyboard: set program (1-9) and manual trigger (t) -
            # In PLC-test sub-mode ('b'), 1-9 writes program_no_test_device and
            # 't' pulses trigger_test_device instead of affecting the scan flow.
            manual_trigger = False
            if args.debug:
                if key == ord('b'):
                    plc_test_mode = not plc_test_mode
                    if not plc_test_mode:
                        plc.write_bit(plc_cfg['trigger_test_device'], 0)
                    print(f"\n[DEBUG] PLC test mode {'ON' if plc_test_mode else 'OFF'}")
                elif ord('1') <= key <= ord('9'):
                    n = key - ord('0')
                    if plc_test_mode:
                        plc.write_word(plc_cfg['program_no_test_device'], n)
                        print(f"\n[PLC-TEST] Wrote program no. {n} -> {plc_cfg['program_no_test_device']}")
                    elif n in detectors:
                        current_program_no = n
                        print(f"\n[DEBUG] Program No. set to {n} ({detectors[n][0]})")
                    else:
                        print(f"\n[DEBUG] No detector for program {n}; ignored")
                elif key == ord('t'):
                    if plc_test_mode:
                        plc.write_bit(plc_cfg['trigger_test_device'], 1)
                        time.sleep(0.1)
                        plc.write_bit(plc_cfg['trigger_test_device'], 0)
                        print(f"\n[PLC-TEST] Pulsed {plc_cfg['trigger_test_device']}")
                    else:
                        manual_trigger = True

            # --- (3) Trigger check --------------------------------------------
            if poll_plc_now:
                trigger_bit = plc.read_bit(plc_cfg['trigger_device'])[0]
                last_plc_poll = time.time()
            if not (trigger_bit == 1 or manual_trigger):
                continue

            # ---- Cycle start ----
            set_status(plc, plc_cfg, ready=0, busy=1, complete=0, error=0)
            plc.write_word(plc_cfg['error_code_device'], ERR_OK)

            program_no = current_program_no
            if program_no is None or program_no not in detectors:
                print(f"\n[ERROR] Cannot scan: program no. is {program_no}")
                plc.write_word(plc_cfg['error_code_device'], ERR_INVALID_PROGRAM)
                set_status(plc, plc_cfg, ready=1, busy=0, complete=0, error=1)
                _wait_trigger_low(plc, plc_cfg)
                continue

            print(f"\n[TRIGGER] Program No. = {program_no}{'  (manual)' if manual_trigger else ''}")
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

            detected_pixels, detected_names, confidences, detected_homos, _ = detector.detect(
                scan_frame, config['camera']['resolution_width'], config['camera']['resolution_height']
            )
            best = best_per_point(detector, detected_names, confidences)

            # ---- Trigger result window: one tile per sub-template -----------
            best_indices = {entry['idx'] for entry in best.values()}
            result_img = _build_trigger_result_grid(
                scan_frame, detected_pixels, detected_names, confidences,
                detected_homos, detector, best_indices,
                header=f"TRIGGER  Program {program_no} ({prog_name})",
            )
            cv2.imshow("Trigger Result", result_img)
            cv2.waitKey(1)  # repaint immediately

            if not best:
                print("\n[WARNING] No targets found.")
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

            # show_3d=False so PLC writes & prints happen immediately
            extracted_6dof = transformer.extract_3d_data(
                filtered_pixels, filtered_names,
                show_3d=False,
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


    except KeyboardInterrupt:
        print("\n[INFO] Exiting program...")
    finally:
        set_status(plc, plc_cfg, ready=0, busy=0, complete=0, error=1)
        plc.disconnect()
        cam.release()
        cv2.destroyAllWindows()


def _build_trigger_result_grid(scan_frame, pixels, names, confs, homos, detector,
                               best_indices, header="", tile_w=280, tile_h=220,
                               crop_w=120, crop_h=100, cols=4):
    """One close-up tile per detected sub-template (PointA.1, PointA.2, ...).
    Best-per-point tiles get a thick green border + 'BEST' label.
    """
    h_frame, w_frame = scan_frame.shape[:2]
    header_h = 36 if header else 0

    tiles = []
    for idx, (pixel, name, conf, homo) in enumerate(zip(pixels, names, confs, homos)):
        px, py = pixel
        x0 = max(0, px - crop_w); y0 = max(0, py - crop_h)
        x1 = min(w_frame, px + crop_w); y1 = min(h_frame, py + crop_h)
        crop = scan_frame[y0:y1, x0:x1].copy()
        if crop.size == 0:
            continue

        sx = tile_w / crop.shape[1]
        sy = tile_h / crop.shape[0]
        tile = cv2.resize(crop, (tile_w, tile_h))

        local_poly = homo.copy()
        local_poly[:, 0, 0] = (local_poly[:, 0, 0] - x0) * sx
        local_poly[:, 0, 1] = (local_poly[:, 0, 1] - y0) * sy
        is_best = idx in best_indices
        color = (0, 255, 0) if is_best else (0, 200, 220)
        thickness = 3 if is_best else 1

        cv2.polylines(tile, [np.int32(local_poly)], True, color, thickness, cv2.LINE_AA)
        cv2.circle(tile, (int((px - x0) * sx), int((py - y0) * sy)), 5, (0, 0, 255), -1)

        meta = detector.templates_by_target.get(name)
        point = meta['point'] if meta else "?"
        title = f"BEST {point}: {name}" if is_best else f"{name}"
        cv2.putText(tile, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        cv2.putText(tile, f"Conf: {conf:.1f}%", (8, tile_h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(tile, (0, 0), (tile_w - 1, tile_h - 1), color, 2 if is_best else 1)
        tiles.append(tile)

    if not tiles:
        canvas = np.zeros((max(tile_h, 80) + header_h, tile_w * cols, 3), dtype=np.uint8)
        cv2.putText(canvas, "No sub-templates matched.", (15, header_h + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        n = len(tiles)
        c = min(cols, n)
        r = (n + c - 1) // c
        grid = np.zeros((r * tile_h, c * tile_w, 3), dtype=np.uint8)
        for i, t in enumerate(tiles):
            rr, cc = i // c, i % c
            grid[rr * tile_h:(rr + 1) * tile_h, cc * tile_w:(cc + 1) * tile_w] = t
        canvas = np.zeros((header_h + grid.shape[0], grid.shape[1], 3), dtype=np.uint8)
        canvas[header_h:, :] = grid

    if header:
        cv2.putText(canvas, header, (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return canvas


def _wait_trigger_low(plc, plc_cfg, timeout_sec=10.0, poll_sec=0.1):
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
