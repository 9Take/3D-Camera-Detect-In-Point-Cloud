import pymcprotocol
import time

class PLCCommunicator:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.plc = pymcprotocol.Type3E()
        self.plc.setaccessopt(commtype="binary")
        self.connected = False

    def connect(self):
        try:
            print(f"[PLC] Connecting to {self.ip}:{self.port}...")
            self.plc.connect(self.ip, self.port)
            self.connected = True
            print("[PLC] Connected Successfully!")
            return True
        except Exception as e:
            print(f"[PLC ERROR] Failed to connect: {e}")
            self.connected = False
            return False

    def disconnect(self):
        if self.connected:
            self.plc.close()
            self.connected = False
            print("[PLC] Disconnected.")

    def read_bit(self, device):
        """อ่านค่า Bit Register (เช่น M100) คืนค่าเป็น List เช่น [0] หรือ [1]"""
        if not self.connected: return [0]
        try:
            return self.plc.batchread_bitunits(device, 1)
        except Exception as e:
            print(f"[PLC ERROR] Read Bit {device} Failed: {e}")
            return [0]

    def write_bit(self, device, value):
        """เขียนค่า Bit Register (0 หรือ 1)"""
        if not self.connected: return
        try:
            self.plc.batchwrite_bitunits(device, [value])
        except Exception as e:
            print(f"[PLC ERROR] Write Bit {device} Failed: {e}")

    def write_scaled_word(self, device, float_val, multiplier=1000):
        """
        แปลงค่า Float (เมตร/องศา) เป็น Int 16-bit แล้วส่งเข้า PLC
        เช่น 0.286 m * 1000 = 286 mm
        """
        if not self.connected: return
        try:
            scaled_val = int(round(float_val * multiplier))
            # ป้องกันค่าล้น (16-bit signed integer)
            if scaled_val > 32767: scaled_val = 32767
            if scaled_val < -32768: scaled_val = -32768
            
            self.plc.batchwrite_wordunits(device, [scaled_val])
        except Exception as e:
            print(f"[PLC ERROR] Write Word {device} Failed: {e}")