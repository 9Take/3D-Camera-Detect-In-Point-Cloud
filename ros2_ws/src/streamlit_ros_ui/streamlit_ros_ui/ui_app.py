import streamlit as st
import pymcprotocol

# --- 1. ปรับแต่ง UI ให้เป็นแบบเรียบๆ ขาว-ดำ สไตล์ PLC/HMI ---
st.set_page_config(page_title="PLC UI", layout="centered")
st.markdown("""
    <style>
    /* บังคับพื้นหลังสีขาว ตัวอักษรสีดำ และใช้ฟอนต์ดูเรียบๆ */
    .stApp {
        background-color: #FFFFFF;
        color: #000000;
        font-family: 'Courier New', Courier, monospace;
    }
    /* ซ่อนเมนูขวาบนและ footer ของ Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    /* ปรับแต่งปุ่มให้ดูแข็งๆ เหมือนปุ่มบน HMI */
    .stButton>button {
        background-color: #E0E0E0;
        color: black;
        border: 2px solid #000000;
        border-radius: 0px;
        width: 100%;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #CCCCCC;
        border: 2px solid #000000;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. ฟังก์ชันสำหรับคุยกับ PLC ---
def send_to_plc(ip, port, register, value):
    try:
        # สร้าง instance สำหรับ MC Protocol (Type 3E นิยมใช้ใน Mitsubishi)
        pymc3e = pymcprotocol.Type3E()
        # ตั้งค่าการสื่อสารเป็นแบบ binary
        pymc3e.setaccessopt(commtype="binary")
        
        # เชื่อมต่อ TCP/IP
        pymc3e.connect(ip, int(port))
        
        # เขียนค่าลง PLC (สมมติว่าเป็น Data Register ระดับ Word เช่น D100)
        # รับค่าเป็น List ของ Integer
        pymc3e.batchwrite_wordunits(headdevice=register, values=[int(value)])
        
        # ปิดการเชื่อมต่อ
        pymc3e.close()
        return True, f"SUCCESS: WROTE {value} TO {register}"
    except Exception as e:
        return False, f"ERROR: {str(e)}"

# --- 3. ส่วนสร้าง UI ---
def main():
    st.subheader("PLC PARAMETER INPUT")
    st.markdown("---")

    # รับค่า Network
    col1, col2 = st.columns([3, 1])
    with col1:
        plc_ip = st.text_input("PLC IP ADDRESS", value="192.168.0.10")
    with col2:
        plc_port = st.text_input("PORT", value="5000")

    # รับค่า Register และข้อมูล
    col3, col4 = st.columns([1, 1])
    with col3:
        register_addr = st.text_input("REGISTER (e.g. D100)", value="D100")
    with col4:
        data_value = st.text_input("VALUE (INT)", value="0")

    st.markdown("<br>", unsafe_allow_html=True)

    # ปุ่มกดส่งค่า
    if st.button("SEND TO PLC"):
        if data_value.lstrip('-').isdigit(): # เช็คว่าเป็นตัวเลข
            success, msg = send_to_plc(plc_ip, plc_port, register_addr, data_value)
            if success:
                st.success(msg)
            else:
                st.error(msg)
        else:
            st.error("ERROR: VALUE MUST BE INTEGER")

if __name__ == "__main__":
    main()