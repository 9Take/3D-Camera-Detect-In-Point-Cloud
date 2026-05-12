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

def parse_device(device_str: str) -> tuple[str, int]:
    """แยกตัวอักษรและตัวเลขออกจากกัน เช่น 'D1000' -> ('D', 1000)"""
    if not device_str or len(device_str) < 2:
        return "D", 0
    prefix = device_str[0].upper()
    try:
        addr = int(device_str[1:])
    except ValueError:
        addr = 0
    return prefix if prefix in ["M", "D"] else "D", addr

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

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_plc_conn, tab_plc_devices, tab_targets, tab_preview = st.tabs([
    "🔌 PLC Connection",
    "📋 PLC Devices (Optimized)",
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
        # Allow empty string for gateway, but validate if user types something
        if gw_val.strip() != "" and not is_valid_ip(gw_val):
            st.warning("⚠️ รูปแบบ Gateway ไม่ถูกต้อง")
            errors.append("รูปแบบ Gateway ไม่ถูกต้อง")
        else:
            plc["gateway"] = gw_val.strip()

# ── PLC Devices (Optimized Selection) ───────────────────────────────────────
with tab_plc_devices:
    st.header("ตั้งค่าอุปกรณ์ PLC (M / D)")
    st.info("ช่วงที่รองรับ: M0-M61439 และ D0-D61439")

    def optimized_device_input(label: str, key: str, default: str):
        current_val = str(plc.get(key, default))
        prefix, addr = parse_device(current_val)
        
        col_pre, col_num = st.columns([1, 3])
        with col_pre:
            new_prefix = st.selectbox("Type", ["M", "D"], index=0 if prefix == "M" else 1, key=f"pre_{key}")
        with col_num:
            new_addr = st.number_input(f"{label} (0-61439)", 0, 61439, addr, key=f"num_{key}")
        
        final_device = f"{new_prefix}{new_addr}"
        plc[key] = final_device
        return final_device

    plc = cfg.setdefault("plc", {})
    
    st.markdown("##### Communication / Handshake")
    optimized_device_input("Heartbeat Register", "heartbeat_device", "D1000")
    optimized_device_input("Error Code Register", "error_device", "D1100")

    st.markdown("---")
    st.markdown("##### Trigger / Status Bits")
    optimized_device_input("Trigger Bit (PLC → PC)", "trigger_device", "M1000")
    optimized_device_input("Status / ACK Bit (PC → PLC)", "status_device", "M1001")

# ── Target Registers (D Only) ────────────────────────────────────────────────
with tab_targets:
    st.header("Target Output Registers")
    st.info("📌 ตำแหน่งของเป้าหมายรองรับเฉพาะ **D Register (D0 - D61439)** เท่านั้น หากพิมพ์ค่าเกิน ระบบจะปรับเป็น 61439 ให้อัตโนมัติ")
    
    targets: dict = plc.setdefault("targets", {})

    for tname, tdata in targets.items():
        with st.expander(f"🎯 Target **{tname}**", expanded=True):
            fields = ["Input_X", "Input_Y", "Input_Z", "Input_r", "Input_p", "Input_y"]
            cols = st.columns(3)
            
            for i, field in enumerate(fields):
                with cols[i % 3]:
                    current_val = tdata.get(field, "D2000")
                    _, addr = parse_device(current_val) 
                    
                    st.write(f"**{field}**")
                    c1, c2 = st.columns([1, 5])
                    with c1:
                        st.markdown("<h4 style='text-align: center; margin-top: 5px;'>D</h4>", unsafe_allow_html=True)
                    with c2:
                        a = st.number_input(
                            "Address", 
                            min_value=0, 
                            max_value=61439, 
                            value=addr, 
                            key=f"num_{tname}_{field}",
                            label_visibility="collapsed" 
                        )
                    tdata[field] = f"D{a}"

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