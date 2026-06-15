# 👋 Handover Guide — 3D Guidance Robot

> This folder explains how the whole system works, broken
> down by area, so you can keep developing it without me. Read this page first, then
> jump to the area you need.

---

## What this system does (in one paragraph)

A PLC tells us "scan now, using model/program number N". We grab an RGB-D frame from an
Intel RealSense camera, find pre-taught features (templates) in the color image with
SIFT matching, look up the depth at each found point, lift it to a 3D position + surface
orientation, transform that from the **camera frame** into the **robot base frame** using
the hand-eye calibration, and write the resulting 6-DOF poses (X, Y, Z, A, B, C) back to
PLC registers so the robot arm can go there. Communication with the PLC is Mitsubishi
**MC Protocol (Type3E binary)** over TCP.

```
PLC ──"scan, program N"──►  PC (this code)  ──X,Y,Z,A,B,C per point──►  PLC ──► Robot
 ▲                              │
 └──── status/heartbeat ───  RealSense D435i (RGB + Depth)
```

---

## The handover docs (read in this order)

| # | Doc | What's inside |
|---|-----|---------------|
| 1 | [getting-started.md](getting-started.md) | Install, run, Docker, debug keyboard controls, how to research the code |
| 2 | [core.md](core.md) | The vision brain: `core/detector.py` (2D find) + `core/transformer.py` (2D→3D pose) |
| 3 | [communication.md](communication.md) | `communication/realsense.py` (camera) + `communication/plc_comm.py` (PLC I/O) + register map |
| 4 | [calibration.md](calibration.md) | Hand-eye calibration: why, how to run `calibration/aruco_calibate.py`, how to verify |
| 5 | [tools.md](tools.md) | Helper scripts: making templates, decoding PLC words, board detect, geometry, offline solver |
| 6 | [configuration.md](configuration.md) | Every key in `config.yaml` explained |
| 7 | [main-loop.md](main-loop.md) | How `main.py` ties everything together, one trigger cycle step by step |

There is also a deeper, code-line-referenced write-up in
[../methodology.md](../methodology.md) — use it when you need exact line numbers.

---

## Project map (where things live)

```
3D-Camera-Detect-In-Point-Cloud/
├── main.py                 ← runtime entry point. THIS is what runs in production.
├── config.yaml             ← all settings (camera, PLC registers, programs, calibration)
├── requirements.txt        ← Python deps (Python 3.8)
├── Dockerfile / docker-compose.yml
│
├── core/
│   ├── detector.py         ← SIFT template matching → 2D pixel + confidence
│   └── transformer.py      ← pixel + depth → 3D point + 6-DOF orientation
│
├── communication/
│   ├── realsense.py        ← RealSense camera wrapper (aligned RGB-D, depth filters)
│   └── plc_comm.py         ← MC-protocol read/write, reconnect, heartbeat, slot writes
│
├── calibration/
│   ├── aruco_calibate.py   ← hand-eye calibration capture + solve (run occasionally)
│   ├── hand_eye_result.npz ← saved calibration result + raw poses
│   └── capture_log.csv     ← raw per-pose log (for offline debugging)
│
├── tools/
│   ├── create_template.py  ← teach a new template (point-and-click on camera feed)
│   ├── plc_decode.py       ← encode/decode int32 ↔ PLC words (pure logic, testable)
│   ├── board_detect.py     ← ChArUco board pose (used by calibration)
│   ├── geometry.py         ← KUKA angle ↔ matrix maths (pure numpy)
│   └── cal_ressult_calib.py← offline brute-force solver to debug a bad calibration
│
├── data/
│   ├── templates/          ← taught templates, organized ProgramX/PointY/*.png + meta.json
│   └── logs/               ← position_mem.json (last scan), current_detect.json
│
├── docs/                   ← you are here
└── memory/                 ← notes on two hard-won bugs (READ THESE, see below)
```

---

## ⚠️ Two bugs that cost days — read before touching calibration

These are in [../../memory/](../../memory/) and are repeated in [calibration.md](calibration.md):

1. **OpenCV 4.7.0 `CharucoBoard.matchImagePoints()` is broken.** It scrambles the
   point pairing and gives garbage board poses (~1400 px reprojection error). We bypass
   it and build correspondences from `getChessboardCorners()`. Don't "fix" it back.

2. **Use `PARK`, not `TSAI`, for hand-eye solving.** Our capture poses have large
   rotations between them, where TSAI's solver is numerically fragile (gave 351 mm error
   vs PARK's 29 mm on the *same* good data). The solver was the problem, not the data.

---

## How to research this project when you're stuck

1. **Start at the data flow**, not a file. Read [main-loop.md](main-loop.md) — it walks
   one trigger from PLC bit to PLC result. Every other module hangs off that.
2. **Run it in `--debug` mode without a robot.** You can drive triggers from the keyboard
   and watch the live window + 3D viewer. See [getting-started.md](getting-started.md).
3. **The PLC side is just registers.** The whole contract with the PLC is the
   `plc:` section of `config.yaml`. If something is "not talking to the PLC", that table
   is the first thing to check — see [communication.md](communication.md).
4. **Use the logs.** `data/logs/position_mem.json` is the last scan's full result.
   `calibration/capture_log.csv` is every calibration pose. Both are made for offline
   debugging.
5. **The `memory/` folder** records non-obvious gotchas discovered the hard way.
