"""
Streamlit-based config.yaml editor for the 3D Camera / PLC detection system.
Run with: streamlit run config_editor.py
"""

from __future__ import annotations

import streamlit as st
import yaml
import copy
import re
from pathlib import Path
import os

current_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_path)  # Go up one level to project root
file_in_root = os.path.join(project_root, "config.yaml")
CONFIG_PATH = Path(file_in_root)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def save_config(cfg: dict):
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def is_valid_ip(ip: str) -> bool:
    pattern = r"^(\d{1,3}\.){3}\d{1,3}$"
    if not re.match(pattern, ip):
        return False
    return all(0 <= int(part) <= 255 for part in ip.split("."))


def is_valid_device(device: str) -> bool:
    """Basic check: letter prefix + digits, e.g. D1000, M1001."""
    return bool(re.match(r"^[A-Za-z]+\d+$", device.strip()))


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="System Config Editor",
    page_icon="⚙️",
    layout="wide",
)

st.title("⚙️  System Configuration Editor")
st.caption(f"Editing: `{CONFIG_PATH.resolve()}`")

# Load once per session, or reload when user requests it
if st.session_state.get("cfg") is None or st.button("🔄 Reload from file", help="Discard unsaved changes and reload config.yaml"):
    try:
        st.session_state.cfg = load_config()
        st.session_state.load_error = None
    except Exception as e:
        st.session_state.load_error = str(e)

if st.session_state.get("load_error"):
    st.error(f"Failed to load config.yaml: {st.session_state.get('load_error')}")
    st.stop()

cfg: dict = st.session_state.get("cfg", {})

errors: list[str] = []

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_cam, tab_plc_conn, tab_plc_devices, tab_targets, tab_paths, tab_preview = st.tabs([
    "📷  Camera",
    "🔌  PLC Connection",
    "📋  PLC Devices",
    "🎯  Target Registers",
    "📁  Paths",
    "🗒️  YAML Preview",
])

# ── Camera ──────────────────────────────────────────────────────────────────
with tab_cam:
    st.header("Camera Resolution")
    st.info("Sets the capture resolution for the Intel RealSense depth camera.")

    cam = cfg.setdefault("camera", {})

    col1, col2 = st.columns(2)
    with col1:
        cam["resolution_width"] = st.number_input(
            "Width (px)",
            min_value=320, max_value=1920, step=8,
            value=int(cam.get("resolution_width", 848)),
            help="Horizontal resolution. Common values: 640, 848, 1280.",
        )
    with col2:
        cam["resolution_height"] = st.number_input(
            "Height (px)",
            min_value=240, max_value=1080, step=8,
            value=int(cam.get("resolution_height", 480)),
            help="Vertical resolution. Common values: 480, 720.",
        )

    st.markdown("**Common presets**")
    preset_cols = st.columns(3)
    if preset_cols[0].button("640 × 480"):
        cam["resolution_width"], cam["resolution_height"] = 640, 480
        st.rerun()
    if preset_cols[1].button("848 × 480"):
        cam["resolution_width"], cam["resolution_height"] = 848, 480
        st.rerun()
    if preset_cols[2].button("1280 × 720"):
        cam["resolution_width"], cam["resolution_height"] = 1280, 720
        st.rerun()

# ── PLC Connection ───────────────────────────────────────────────────────────
with tab_plc_conn:
    st.header("PLC Network Connection")
    st.info(
        "Connection settings for the Mitsubishi PLC using the MC Protocol (3E frame).\n\n"
        "Make sure the PC and PLC are on the **same network subnet** (e.g. 192.168.1.x)."
    )

    plc = cfg.setdefault("plc", {})

    col1, col2 = st.columns([3, 1])
    with col1:
        ip_val = st.text_input(
            "PLC IP Address",
            value=str(plc.get("ip", "192.168.1.165")),
            placeholder="192.168.1.165",
            help="IPv4 address of the PLC (e.g. 192.168.1.165).",
        )
        if not is_valid_ip(ip_val):
            st.warning("⚠️ Invalid IP address format.")
            errors.append("PLC IP address is invalid.")
        else:
            plc["ip"] = ip_val

    with col2:
        plc["port"] = st.number_input(
            "Port",
            min_value=1, max_value=65535, step=1,
            value=int(plc.get("port", 5010)),
            help="MC Protocol port on the PLC (default 5010).",
        )

    st.markdown("---")
    st.subheader("Point Data Settings")

    col3, col4 = st.columns(2)
    with col3:
        plc["registers_per_point"] = st.number_input(
            "Registers per point",
            min_value=1, max_value=32, step=1,
            value=int(plc.get("registers_per_point", 6)),
            help="Number of word registers used per detected point (X, Y, Z, Roll, Pitch, Yaw = 6).",
        )
    with col4:
        plc["max_points"] = st.number_input(
            "Maximum points to send",
            min_value=1, max_value=20, step=1,
            value=int(plc.get("max_points", 5)),
            help="Maximum number of detected targets that will be sent to the PLC per trigger.",
        )

# ── PLC Devices ─────────────────────────────────────────────────────────────
with tab_plc_devices:
    st.header("PLC Device (Register) Addresses")
    st.info(
        "Enter the PLC register/device addresses used for handshake and status signals.\n"
        "Format: **letter prefix + number**, e.g. `D1000`, `M1001`."
    )

    plc = cfg.setdefault("plc", {})

    def device_input(label: str, key: str, default: str, help_text: str) -> str:
        val = st.text_input(label, value=str(plc.get(key, default)), help=help_text, key=f"dev_{key}")
        if not is_valid_device(val):
            st.warning(f"⚠️ '{val}' does not look like a valid device address.")
            errors.append(f"{label} has an invalid device address.")
        return val

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Communication / Handshake")
        plc["heartbeat_device"] = device_input(
            "Heartbeat Register", "heartbeat_device", "D1000",
            "Word register that the PC writes periodically so the PLC can detect a PC disconnect.",
        )
        plc["error_device"] = device_input(
            "Error Code Register", "error_device", "D1100",
            "Word register where error codes are written when a fault occurs.",
        )

    with col2:
        st.markdown("##### Trigger / Status Bits")
        plc["trigger_device"] = device_input(
            "Trigger Bit (PLC → PC)", "trigger_device", "M1000",
            "Bit set by the PLC to request a measurement cycle.",
        )
        plc["status_device"] = device_input(
            "Status / ACK Bit (PC → PLC)", "status_device", "M1001",
            "Bit set by the PC to acknowledge that data has been written (handshake).",
        )

    st.markdown("---")
    st.markdown("##### Result Data")
    plc["point_count_device"] = device_input(
        "Point Count Register", "point_count_device", "D1101",
        "Word register that receives the number of detected targets found.",
    )

# ── Target Registers ─────────────────────────────────────────────────────────
with tab_targets:
    st.header("Target Output Registers")
    st.info(
        "Each **target** (A, B, …) needs six consecutive word registers for the 6-DOF pose data:\n"
        "X, Y, Z (position in mm × 10 000) and Roll, Pitch, Yaw (radians × 10 000)."
    )

    plc = cfg.setdefault("plc", {})
    targets: dict = plc.setdefault("targets", {})

    # Add / remove targets
    with st.expander("➕ Add or remove a target"):
        new_name = st.text_input("New target name (single letter, e.g. C)", max_chars=4)
        add_col, del_col = st.columns(2)
        if add_col.button("Add target") and new_name:
            key = new_name.strip().upper()
            if key not in targets:
                # Auto-fill placeholder registers
                targets[key] = {
                    "Input_X": "", "Input_Y": "", "Input_Z": "",
                    "Input_r": "", "Input_p": "", "Input_y": "",
                }
                st.success(f"Target '{key}' added. Fill in its register addresses below.")
                st.rerun()
            else:
                st.warning(f"Target '{key}' already exists.")

        remove_key = del_col.selectbox("Remove target", options=["—"] + list(targets.keys()))
        if del_col.button("Remove") and remove_key != "—":
            targets.pop(remove_key, None)
            st.rerun()

    DOF_LABELS = {
        "Input_X": ("X Position", "Word register for X coordinate (metres × 10 000)"),
        "Input_Y": ("Y Position", "Word register for Y coordinate (metres × 10 000)"),
        "Input_Z": ("Z Position", "Word register for Z coordinate (metres × 10 000)"),
        "Input_r": ("Roll",       "Word register for Roll angle (radians × 10 000)"),
        "Input_p": ("Pitch",      "Word register for Pitch angle (radians × 10 000)"),
        "Input_y": ("Yaw",        "Word register for Yaw angle (radians × 10 000)"),
    }

    for tname, tdata in targets.items():
        with st.expander(f"🎯  Target  **{tname}**", expanded=True):
            cols = st.columns(3)
            for i, (field, (label, tip)) in enumerate(DOF_LABELS.items()):
                with cols[i % 3]:
                    val = st.text_input(
                        label,
                        value=str(tdata.get(field, "")),
                        help=tip,
                        key=f"target_{tname}_{field}",
                    )
                    if val and not is_valid_device(val):
                        st.warning(f"⚠️ Invalid: {val}")
                        errors.append(f"Target {tname} / {label}: invalid device address.")
                    tdata[field] = val

# ── Paths ────────────────────────────────────────────────────────────────────
with tab_paths:
    st.header("File & Folder Paths")
    st.info(
        "Paths are **relative to the project root** (where `main.py` lives). "
        "Use forward slashes `/` even on Windows — Python handles the conversion."
    )

    paths = cfg.setdefault("paths", {})

    paths["save_dir"] = st.text_input(
        "Output / Save Directory",
        value=str(paths.get("save_dir", "data/templates/model")),
        help="Where 3D scans and detection results are saved.",
    )
    paths["template_dir"] = st.text_input(
        "Template Directory",
        value=str(paths.get("template_dir", "data/templates/model")),
        help="Folder containing the 2D template images used for detection.",
    )
    paths["position_mem"] = st.text_input(
        "Position Memory File (JSON)",
        value=str(paths.get("position_mem", "data/logs/position_mem.json")),
        help="JSON file where the last known target positions are persisted between runs.",
    )

# ── YAML Preview ─────────────────────────────────────────────────────────────
with tab_preview:
    st.header("Current YAML Preview")
    st.caption("This is what will be written to config.yaml when you click Save.")
    yaml_text = yaml.dump(cfg, default_flow_style=False, allow_unicode=True, sort_keys=False)
    st.code(yaml_text, language="yaml")

# ---------------------------------------------------------------------------
# Save button (always visible at the bottom)
# ---------------------------------------------------------------------------

st.divider()

save_col, status_col = st.columns([1, 3])

with save_col:
    save_clicked = st.button("💾  Save Configuration", type="primary", use_container_width=True)

with status_col:
    if errors:
        st.error("Fix the following issues before saving:\n" + "\n".join(f"• {e}" for e in errors))

if save_clicked:
    if errors:
        st.error("Configuration was NOT saved because of the errors above.")
    else:
        try:
            save_config(cfg)
            st.success(f"✅  Configuration saved to `{CONFIG_PATH.resolve()}`")
        except Exception as exc:
            st.error(f"Failed to write file: {exc}")