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

def read_plc_value(client, addr: str) -> str:
    """ฟังก์ชันช่วยอ่านค่า PLC และแปลงเป็น String สำหรับแสดงผล"""
    if not addr or client is None:
        return "N/A"
    prefix = addr[0].upper()
    try:
        if prefix == 'M':
            val = client.read_bit(addr)
            return str(val[0]) if val else "Error"
        elif prefix == 'D':
            if hasattr(client, 'read_word'):
                val = client.read_word(addr)
                return str(val[0]) if val else "Error"
            else:
                return "N/A (No read_word)"
    except Exception:
        return "Error"
    return "N/A"

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
# Navigation (ใช้ Radio แนวนอนแทน Tabs เพื่อแก้ปัญหาการเด้งกลับหน้าแรก)
# ---------------------------------------------------------------------------
st.write("---")
menu_options = [
    "🔌 PLC Connection",
    "📋 PLC Devices (M/D)",
    "🎯 Target Registers",
    "📡 PLC Monitor (Live)",
    "🗒️ YAML Preview"
]
active_tab = st.radio("Navigation Menu", menu_options, horizontal=True, label_visibility="collapsed")
st.write("---")

# ── PLC Connection ───────────────────────────────────────────────────────────
if active_tab == "🔌 PLC Connection":
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
elif active_tab == "📋 PLC Devices (M/D)":
    st.header("ตั้งค่าอุปกรณ์ PLC (M / D)")
    plc = cfg.setdefault("plc", {})
    
    def combined_device_input(label: str, key: str, default: str):
        current_val = str(plc.get(key, default)).upper()
        idx = options_md.index(current_val) if current_val in options_md else options_md.index(default)
        selected = st.selectbox(label, options=options_md, index=idx, key=f"sel_{key}")
        plc[key] = selected
        return selected

    st.markdown("##### Communication / Handshake")
    combined_device_input("Heartbeat Register", "heartbeat_device", "D1000")
    combined_device_input("Error Code Register", "error_device", "D1100")
    st.markdown("##### Trigger / Status Bits")
    combined_device_input("Trigger Bit (PLC → PC)", "trigger_device", "M1000")
    combined_device_input("Status / ACK Bit (PC → PLC)", "status_device", "M1001")

# ── Target Registers ─────────────────────────────────────────────────────────
elif active_tab == "🎯 Target Registers":
    st.header("Target Output Registers")
    plc = cfg.setdefault("plc", {})
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
elif active_tab == "📡 PLC Monitor (Live)":
    st.header("📡 Live PLC Monitor")
    st.info("ระบบจะดึงค่าจาก Register ทุกตัวที่คุณตั้งไว้ในหน้าก่อนๆ มาแสดงผลพร้อมกัน")
    plc = cfg.setdefault("plc", {})

    if PLCCommunicator is None:
        st.error("ไม่สามารถ import `PLCCommunicator` ได้ ตรวจสอบ Path หรือไฟล์ `communication/plc_comm.py`")
    else:
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

        # ส่วนอ่านค่าแบบแสดงผลทั้งหมด (Auto Read from Config)
        if is_connected:
            col_title, col_btn = st.columns([4, 1])
            with col_title:
                st.subheader("📊 ข้อมูล Device ปัจจุบัน")
            with col_btn:
                do_read = st.button("🔄 ดึงค่าล่าสุด", use_container_width=True)
            
            # โชว์ข้อมูลเมื่อกดปุ่ม (หรือกดดึงค่า)
            if do_read:
                client = st.session_state.plc_client
                
                # 1. แสดงค่า System Devices
                st.markdown("#### ⚙️ System Devices")
                sys_cols = st.columns(4)
                sys_devices = {
                    "Heartbeat": plc.get("heartbeat_device", ""),
                    "Error Code": plc.get("error_device", ""),
                    "Trigger Bit": plc.get("trigger_device", ""),
                    "Status Bit": plc.get("status_device", "")
                }
                
                for i, (name, addr) in enumerate(sys_devices.items()):
                    val = read_plc_value(client, addr)
                    sys_cols[i].metric(label=f"{name} ({addr})", value=val)
                
                st.write("")
                
                # 2. แสดงค่า Target Registers ทั้งหมด
                st.markdown("#### 🎯 Target Registers")
                targets = plc.get("targets", {})
                for tname, tdata in targets.items():
                    st.markdown(f"**{tname}**")
                    t_cols = st.columns(6)
                    fields = ["Input_X", "Input_Y", "Input_Z", "Input_r", "Input_p", "Input_y"]
                    for i, field in enumerate(fields):
                        addr = tdata.get(field, "")
                        val = read_plc_value(client, addr)
                        t_cols[i].metric(label=f"{field} ({addr})", value=val)
            else:
                st.caption("👈 กดปุ่ม 'ดึงค่าล่าสุด' เพื่ออ่านข้อมูลทั้งหมดจาก PLC")

# ── YAML Preview ─────────────────────────────────────────────────────────────
elif active_tab == "🗒️ YAML Preview":
    st.header("ตัวอย่างไฟล์ YAML")
    yaml_text = yaml.dump(cfg, default_flow_style=False, allow_unicode=True, sort_keys=False)
    st.code(yaml_text, language="yaml")

# ---------------------------------------------------------------------------
# Save button (โชว์ทุกหน้า)
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