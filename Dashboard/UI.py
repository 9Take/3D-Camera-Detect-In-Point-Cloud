"""
Streamlit-based config.yaml editor for the 3D Camera / PLC detection system.
Run with: streamlit run config_editor.py
"""

from __future__ import annotations

import streamlit as st
import yaml
import re
from pathlib import Path
import os

current_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_path)
file_in_root = os.path.join(project_root, "config.yaml")
CONFIG_PATH = Path(file_in_root)

# ---------------------------------------------------------------------------
# Helpers & Cache (เพื่อลดความหน่วงตอนโหลดข้อมูลแสนกว่ารายการ)
# ---------------------------------------------------------------------------

@st.cache_data
def get_md_devices() -> list[str]:
    """สร้างลิสต์รายการ M0-M61439 และ D0-D61439 สำหรับเมนู Dropdown"""
    m_list = [f"M{i}" for i in range(61440)]
    d_list = [f"D{i}" for i in range(61440)]
    return m_list + d_list

@st.cache_data
def get_d_devices() -> list[str]:
    """สร้างลิสต์รายการ D0-D61439 (เฉพาะ D) สำหรับ Targets"""
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

st.set_page_config(
    page_title="PLC Config Editor",
    page_icon="⚙️",
    layout="wide",
)

st.title("⚙️  System Configuration Editor")
st.caption(f"กำลังแก้ไขไฟล์: `{CONFIG_PATH.resolve()}`")

if st.session_state.get("cfg") is None or st.button("🔄 โหลดไฟล์ใหม่", help="ยกเลิกการเปลี่ยนแปลงและโหลดจาก config.yaml"):
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

# โหลดตัวเลือกสำหรับ Dropdown ไว้ล่วงหน้า
options_md = get_md_devices()
options_d = get_d_devices()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_plc_conn, tab_plc_devices, tab_targets, tab_preview = st.tabs([
    "🔌 PLC Connection",
    "📋 PLC Devices (M/D)",
    "🎯 Target Registers (D Only)",
    "🗒️ YAML Preview",
])

# ── PLC Connection ───────────────────────────────────────────────────────────
with tab_plc_conn:
    st.header("การเชื่อมต่อเครือข่าย PLC")
    st.info("ตรวจสอบให้แน่ใจว่า PC และ PLC อยู่ในวงเครือข่าย (Subnet) เดียวกัน")
    
    plc = cfg.setdefault("plc", {})
    
    st.subheader("Network Settings")
    col1, col2 = st.columns(2)
    
    with col1:
        ip_val = st.text_input("PLC IP Address", value=str(plc.get("ip", "192.168.1.165")))
        if not is_valid_ip(ip_val):
            st.warning("⚠️ รูปแบบ IP Address ไม่ถูกต้อง")
            errors.append("รูปแบบ PLC IP Address ไม่ถูกต้อง")
        else:
            plc["ip"] = ip_val

        subnet_val = st.text_input("Subnet Mask", value=str(plc.get("subnet_mask", "255.255.255.0")))
        if not is_valid_ip(subnet_val):
            st.warning("⚠️ รูปแบบ Subnet Mask ไม่ถูกต้อง")
            errors.append("รูปแบบ Subnet Mask ไม่ถูกต้อง")
        else:
            plc["subnet_mask"] = subnet_val

    with col2:
        plc["port"] = st.number_input("Port", 1, 65535, int(plc.get("port", 5010)))
        
        gw_val = st.text_input("Default Gateway", value=str(plc.get("gateway", "")), placeholder="เว้นว่างไว้หากไม่มี Gateway")
        if gw_val.strip() != "" and not is_valid_ip(gw_val):
            st.warning("⚠️ รูปแบบ Gateway ไม่ถูกต้อง")
            errors.append("รูปแบบ Gateway ไม่ถูกต้อง")
        else:
            plc["gateway"] = gw_val.strip()

# ── PLC Devices (Searchable Dropdown) ───────────────────────────────────────
with tab_plc_devices:
    st.header("ตั้งค่าอุปกรณ์ PLC (M / D)")
    st.info("💡 สามารถคลิกแล้วพิมพ์ค้นหาได้เลย (เช่น พิมพ์ D1000)")

    def combined_device_input(label: str, key: str, default: str):
        current_val = str(plc.get(key, default)).upper()
        
        # หา Index ของค่าเดิมใน List (ถ้าไม่มีให้ใช้ค่า Default)
        try:
            idx = options_md.index(current_val)
        except ValueError:
            idx = options_md.index(default) if default in options_md else 0
            
        selected = st.selectbox(label, options=options_md, index=idx, key=f"sel_{key}")
        plc[key] = selected
        return selected

    plc = cfg.setdefault("plc", {})
    
    st.markdown("##### Communication / Handshake")
    combined_device_input("Heartbeat Register", "heartbeat_device", "D1000")
    combined_device_input("Error Code Register", "error_device", "D1100")

    st.markdown("---")
    st.markdown("##### Trigger / Status Bits")
    combined_device_input("Trigger Bit (PLC → PC)", "trigger_device", "M1000")
    combined_device_input("Status / ACK Bit (PC → PLC)", "status_device", "M1001")

# ── Target Registers (D Only - Searchable Dropdown) ──────────────────────────
with tab_targets:
    st.header("Target Output Registers")
    st.info("📌 รองรับเฉพาะ **D Register (D0 - D61439)** สามารถพิมพ์ค้นหาในช่องได้เลย")
    
    targets: dict = plc.setdefault("targets", {})

    for tname, tdata in targets.items():
        with st.expander(f"🎯 Target **{tname}**", expanded=True):
            fields = ["Input_X", "Input_Y", "Input_Z", "Input_r", "Input_p", "Input_y"]
            cols = st.columns(3)
            
            for i, field in enumerate(fields):
                with cols[i % 3]:
                    current_val = str(tdata.get(field, "D2000")).upper()
                    
                    # บังคับให้เป็น D เสมอ (กรณีไฟล์เก่าติดค่า M มา)
                    if not current_val.startswith("D"):
                        current_val = "D2000"
                        
                    try:
                        idx = options_d.index(current_val)
                    except ValueError:
                        idx = options_d.index("D2000")
                        
                    st.write(f"**{field}**")
                    selected = st.selectbox(
                        "Address", 
                        options=options_d, 
                        index=idx, 
                        key=f"sel_{tname}_{field}",
                        label_visibility="collapsed"
                    )
                    tdata[field] = selected

# ── YAML Preview ─────────────────────────────────────────────────────────────
with tab_preview:
    st.header("ตัวอย่างไฟล์ YAML")
    st.caption("ข้อมูลที่จะถูกบันทึกลงใน config.yaml")
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