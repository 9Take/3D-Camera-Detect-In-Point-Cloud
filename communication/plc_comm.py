import pymcprotocol

def float_to_scaled_16bit(val):
    scaled_val = int(round(val * 100))
    if scaled_val > 32767: scaled_val = 32767
    if scaled_val < -32768: scaled_val = -32768
    return [scaled_val]

def send_to_plc(plc_ip, plc_port, start_d_reg, data_A):
    try:
        print(f"\n[PLC] Connecting to {plc_ip}:{plc_port}...")
        plc = pymcprotocol.Type3E()
        plc.setaccessopt(commtype="binary")
        plc.connect(plc_ip, plc_port)
        
        payload = []
        for val in data_A[:3]:
            payload.extend(float_to_scaled_16bit(val))
            
        plc.batchwrite_wordunits(f"D{start_d_reg}", payload)
        
        print(f"[PLC] SUCCESS! Scaled Data sent to D{start_d_reg}-D{start_d_reg+2}")
        plc.close()
    except Exception as e:
        print(f"[PLC ERROR] Failed to send data: {e}")