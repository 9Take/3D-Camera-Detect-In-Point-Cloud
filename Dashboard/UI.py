"""
Streamlit-based config.yaml editor for the 3D Camera / PLC detection system.
Run with: streamlit run config_editor.py
"""

from __future__ import annotations

import streamlit as st
import yaml
import re
import sys
from pathlib import Path
import os

current_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_path)
file_in_root = os.path.join(project_root, "config.yaml")
CONFIG_PATH = Path(file_in_root)

# นำเข้า PLCCommunicator จากโฟลเดอร์โปรเจกต์
sys.path.append(project_root)
try:
    from communication.plc_comm import PLCCommunicator
except ImportError:
    PLCCommunicator = None

# ---------------------------------------------------------------------------
# Helpers & Cache
# ---------------------------------------------------------------------------

@st.cache_data
def get_md_devices() -> list[str]:
    m_list = [f"M{i}" for i in range(61440)]
    d_list = [f"D{i}" for i in range(61440)]
    return m_list + d_list

@st.cache_data
def get_d_devices() -> list[str]:
    return [f"D{i}" for i in range(61440)]

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

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="PLC Config Editor", page_icon="⚙️", layout="wide")

st.title("⚙️  System Configuration Editor")
st.caption(f"กำลังแก้ไขไฟล์: `{CONFIG_PATH.resolve()}`")

# ---------------------------------------------------------------------------
# Session States
# ---------------------------------------------------------------------------
if "plc_client" not in st.session_state:
    st.session_state.plc_client = None

if st.session_state.get("cfg") is None or st.button("🔄 โหลดไฟล์ใหม่", help="โหลด config.yaml ใหม่"):
    try:
        st.session_state.cfg = load_config()
        st.session_state.load_error = None
    except Exception as e:
        st.session_state.load_error = str(e)

if st.session_state.get("load_error"):
    st.error(f"ไม่สามารถโหลดไฟล์ได้: {st.session_state.get('load_error')}")
    st.stop()

cfg: dict = st.session_state.get("cfg", {})
errors: list[str] = []

options_md = get_md_devices()
options_d = get_d_devices()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_plc_conn, tab_plc_devices, tab_targets, tab_monitor, tab_preview = st.tabs([
    "🔌 PLC Connection",
    "📋 PLC Devices (M/D)",
    "🎯 Target Registers",
    "📡 PLC Monitor (Live)",
    "🗒️ YAML Preview",
])

# ── PLC Connection ───────────────────────────────────────────────────────────
with tab_plc_conn:
    st.header("การเชื่อมต่อเครือข่าย PLC")
    plc = cfg.setdefault("plc", {})
    col1, col2 = st.columns(2)
    
    with col1:
        ip_val = st.text_input("PLC IP Address", value=str(plc.get("ip", "192.168.1.165")))
        if not is_valid_ip(ip_val):
            errors.append("รูปแบบ PLC IP Address ไม่ถูกต้อง")
        else:
            plc["ip"] = ip_val

        subnet_val = st.text_input("Subnet Mask", value=str(plc.get("subnet_mask", "255.255.255.0")))
        if not is_valid_ip(subnet_val):
            errors.append("รูปแบบ Subnet Mask ไม่ถูกต้อง")
        else:
            plc["subnet_mask"] = subnet_val

    with col2:
        plc["port"] = st.number_input("Port", 1, 65535, int(plc.get("port", 5010)))
        gw_val = st.text_input("Default Gateway", value=str(plc.get("gateway", "")), placeholder="เว้นว่างไว้หากไม่มี")
        if gw_val.strip() != "" and not is_valid_ip(gw_val):
            errors.append("รูปแบบ Gateway ไม่ถูกต้อง")
        else:
            plc["gateway"] = gw_val.strip()

# ── PLC Devices ──────────────────────────────────────────────────────────────
with tab_plc_devices:
    st.header("ตั้งค่าอุปกรณ์ PLC (M / D)")
    def combined_device_input(label: str, key: str, default: str):
        current_val = str(plc.get(key, default)).upper()
        idx = options_md.index(current_val) if current_val in options_md else options_md.index(default)
        selected = st.selectbox(label, options=options_md, index=idx, key=f"sel_{key}")
        plc[key] = selected
        return selected

    plc = cfg.setdefault("plc", {})
    st.markdown("##### Communication / Handshake")
    combined_device_input("Heartbeat Register", "heartbeat_device", "D1000")
    combined_device_input("Error Code Register", "error_device", "D1100")
    st.markdown("##### Trigger / Status Bits")
    combined_device_input("Trigger Bit (PLC → PC)", "trigger_device", "M1000")
    combined_device_input("Status / ACK Bit (PC → PLC)", "status_device", "M1001")

# ── Target Registers ─────────────────────────────────────────────────────────
with tab_targets:
    st.header("Target Output Registers")
    targets: dict = plc.setdefault("targets", {})

    for tname, tdata in targets.items():
        with st.expander(f"🎯 Target **{tname}**", expanded=True):
            fields = ["Input_X", "Input_Y", "Input_Z", "Input_r", "Input_p", "Input_y"]
            cols = st.columns(3)
            for i, field in enumerate(fields):
                with cols[i % 3]:
                    current_val = str(tdata.get(field, "D2000")).upper()
                    if not current_val.startswith("D"): current_val = "D2000"
                    idx = options_d.index(current_val) if current_val in options_d else options_d.index("D2000")
                        
                    st.write(f"**{field}**")
                    selected = st.selectbox("Address", options=options_d, index=idx, key=f"sel_{tname}_{field}", label_visibility="collapsed")
                    tdata[field] = selected

# ── PLC Monitor (Live Read) ──────────────────────────────────────────────────
with tab_monitor:
    st.header("📡 Live PLC Monitor")
    st.info("เชื่อมต่อกับ PLC และทดลองอ่านค่า Register (M และ D) แบบ Real-time")

    if PLCCommunicator is None:
        st.error("ไม่สามารถ import `PLCCommunicator` ได้ ตรวจสอบ Path หรือไฟล์ `communication/plc_comm.py`")
    else:
        # แผงควบคุมการเชื่อมต่อ
        conn_col, status_col = st.columns([1, 2])
        
        is_connected = st.session_state.plc_client is not None and st.session_state.plc_client.connected
        
        with conn_col:
            if not is_connected:
                if st.button("🔌 Connect PLC", type="primary", use_container_width=True):
                    client = PLCCommunicator(plc.get("ip"), int(plc.get("port")))
                    if client.connect():
                        st.session_state.plc_client = client
                        st.rerun()
                    else:
                        st.error("การเชื่อมต่อล้มเหลว ตรวจสอบ IP/Port หรือสายแลน")
            else:
                if st.button("❌ Disconnect", type="secondary", use_container_width=True):
                    st.session_state.plc_client.disconnect()
                    st.session_state.plc_client = None
                    st.rerun()
                    
        with status_col:
            if is_connected:
                st.success(f"✅ Connected to {plc.get('ip')}:{plc.get('port')}")
            else:
                st.warning("⚠️ Disconnected")

        st.divider()

        # ส่วนอ่านค่า
        if is_connected:
            st.subheader("อ่านค่า Register ปัจจุบัน")
            read_col1, read_col2, read_col3 = st.columns([2, 1, 3])
            
            with read_col1:
                test_device = st.selectbox("เลือก Address ที่ต้องการอ่าน", options=options_md, key="test_device")
                
            with read_col2:
                st.write("")
                st.write("")
                do_read = st.button("📥 Read Value", use_container_width=True)
                
            with read_col3:
                if do_read:
                    st.write("")
                    st.write("")
                    prefix = test_device[0]
                    client = st.session_state.plc_client
                    
                    if prefix == 'M':
                        val = client.read_bit(test_device)
                        st.info(f"**{test_device}** = `{val[0]}` (Bit)")
                    elif prefix == 'D':
                        # เรียกใช้งาน read_word (ต้องแน่ใจว่าเพิ่มฟังก์ชันนี้ใน plc_comm.py แล้ว)
                        if hasattr(client, 'read_word'):
                            val = client.read_word(test_device)
                            st.info(f"**{test_device}** = `{val[0]}` (Word 16-bit)")
                        else:
                            st.error("ไม่พบฟังก์ชัน `read_word` ในคลาส PLCCommunicator")

# ── YAML Preview ─────────────────────────────────────────────────────────────
with tab_preview:
    st.header("ตัวอย่างไฟล์ YAML")
    yaml_text = yaml.dump(cfg, default_flow_style=False, allow_unicode=True, sort_keys=False)
    st.code(yaml_text, language="yaml")

# ---------------------------------------------------------------------------
# Save button
# ---------------------------------------------------------------------------
st.divider()

if errors:
    st.error("กรุณาแก้ไขข้อผิดพลาดต่อไปนี้ก่อนบันทึก:\n" + "\n".join(f"• {e}" for e in errors))

if st.button("💾 บันทึกการตั้งค่า", type="primary", use_container_width=True, disabled=bool(errors)):
    try:
        save_config(cfg)
        st.success(f"✅ บันทึกสำเร็จที่ `{CONFIG_PATH.resolve()}`")
    except Exception as exc:
        st.error(f"เกิดข้อผิดพลาดในการบันทึก: {exc}")