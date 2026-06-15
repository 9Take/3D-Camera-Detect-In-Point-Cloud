# 1. Getting Started

> 🌐 Language: **English** | [ไทย](getting-started.th.md)

How to install, run, and develop the system. Start here on day one.

---

## Hardware you need

- **Intel RealSense D435i** (or any D4xx) plugged into USB 3.0.
- **Mitsubishi PLC** reachable on the network, speaking **MC Protocol Type3E (binary)**.
  Default address `192.168.1.165:5010` (set in `config.yaml`).
- The **robot** (KUKA) — but you do **not** need it to develop the vision side. Use
  `--debug` mode and a PLC simulator, or just the camera.

You can do most development with **only the camera** (template making, detection tuning)
and **only the PLC** (handshake testing). The full loop needs all three.

---

## Install (native, recommended for development)

Python **3.8** is required (RealSense + Open3D wheels are pinned to it).

```bash
cd "3D-Camera-Detect-In-Point-Cloud"
python3.8 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Key dependencies and why:
- `pyrealsense2` — camera driver.
- `opencv-python==4.7.0.72` — SIFT, homography, ArUco. **Pinned**; see the calibration
  bug note. Don't bump it casually.
- `open3d==0.17.0` — point cloud + normal estimation + 3D viewer. (On ARM/Jetson the
  Docker build drops to 0.16.0 because no 0.17 aarch64 wheel exists — APIs used are
  compatible.)
- `pymcprotocol` — Mitsubishi MC protocol client.
- `scipy` — rotation/Euler maths.
- `numpy==1.24.3`, `PyYAML`.

---

## Run

```bash
# Production: wait for the PLC to send program no. + trigger
python main.py

# Development without a robot: keyboard-driven triggers + 3D viewer after each scan
python main.py --debug
```

`main.py` is the **only** runtime entry point. (`main3.py` is an older legacy version —
ignore it.)

### What you'll see
- A **"Vision System - Live"** window: the live camera feed with the best detected target
  per point drawn on it, plus a one-line heartbeat in the console.
- After a trigger, a **"Trigger Result"** grid: one tile per matched sub-template, the
  chosen best per point framed in green.

---

## Keyboard controls (live window must be focused)

Always available:

| Key | Action |
|-----|--------|
| `p` | Open the 3D point-cloud viewer for the **last** scan (non-blocking until you press it) |
| `q` / `ESC` | Quit |

Only in `--debug`:

| Key | Action |
|-----|--------|
| `1`–`9` | Manually select program number |
| `t` | Manual trigger (run a scan now) |
| `b` | Toggle **PLC-test sub-mode** |

In **PLC-test sub-mode** (for a PLC engineer to verify their side of the handshake):

| Key | Action |
|-----|--------|
| `1`–`9` | Write that program no. to `program_no_test_device` (D1500) |
| `t` | Pulse `trigger_test_device` |
| `b` | Exit PLC-test sub-mode |

---

## Run with Docker

`docker-compose.yml` is set up for camera + display passthrough:

```bash
xhost +local:                 # allow the container to open windows on your X server
docker compose up --build
```

It runs `python3 main.py`, uses `network_mode: host` (so it can reach the PLC),
`privileged` + `/dev/bus/usb` (RealSense access), and mounts `./data` and `./config.yaml`
so your templates/config are live-editable without rebuilding.

---

## How to develop safely

This project ships a [`CLAUDE.md`](../../CLAUDE.md) with house rules. The short version:
- **Surgical changes** — touch only what the task needs; match the existing style.
- **Simplicity first** — minimum code that solves the problem.
- **Verify with a goal** — e.g. "run a trigger, confirm slot values in `position_mem.json`".

Good first things to try:
1. `python main.py --debug`, press `1` then `t`, watch a scan run end-to-end (PLC writes
   will just fail quietly if no PLC — that's fine for seeing the vision path).
2. Open `data/logs/position_mem.json` after a scan to see the exact result that was sent.
3. Read [main-loop.md](main-loop.md) alongside `main.py` with both open.
