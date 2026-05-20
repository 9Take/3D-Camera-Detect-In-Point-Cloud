import pymcprotocol
import threading
import time
import re


def _offset_device(device, offset):
    """Return a device address shifted by `offset` words (e.g. D1110 + 3 -> 'D1113')."""
    m = re.match(r"^([A-Za-z]+)(\d+)$", device)
    if not m:
        raise ValueError(f"Invalid device address: {device}")
    prefix, num = m.group(1), int(m.group(2))
    return f"{prefix}{num + offset}"


def _clamp_int16(v):
    if v > 32767: return 32767
    if v < -32768: return -32768
    return int(v)


class PLCCommunicator:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.plc = pymcprotocol.Type3E()
        self.plc.setaccessopt(commtype="binary")
        self.connected = False
        self._lock = threading.Lock()
        self._hb_thread = None
        self._hb_stop = threading.Event()
        self.heartbeat_counter = 0

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
        self.stop_heartbeat()
        if self.connected:
            self.plc.close()
            self.connected = False
            print("[PLC] Disconnected.")

    def read_bit(self, device):
        if not self.connected: return [0]
        try:
            with self._lock:
                return self.plc.batchread_bitunits(device, 1)
        except Exception as e:
            print(f"[PLC ERROR] Read Bit {device} Failed: {e}")
            return [0]

    def read_word(self, device):
        if not self.connected: return 0
        try:
            with self._lock:
                return self.plc.batchread_wordunits(device, 1)[0]
        except Exception as e:
            print(f"[PLC ERROR] Read Word {device} Failed: {e}")
            return 0

    def write_bit(self, device, value):
        if not self.connected: return
        try:
            with self._lock:
                self.plc.batchwrite_bitunits(device, [int(value)])
        except Exception as e:
            print(f"[PLC ERROR] Write Bit {device} Failed: {e}")

    def write_word(self, device, value):
        if not self.connected: return
        try:
            with self._lock:
                self.plc.batchwrite_wordunits(device, [_clamp_int16(value)])
        except Exception as e:
            print(f"[PLC ERROR] Write Word {device} Failed: {e}")

    def write_words(self, device, values):
        """Write a consecutive block of words starting at `device`."""
        if not self.connected or not values: return
        try:
            with self._lock:
                self.plc.batchwrite_wordunits(device, [_clamp_int16(v) for v in values])
        except Exception as e:
            print(f"[PLC ERROR] Write Words {device} (n={len(values)}) Failed: {e}")

    def write_scaled_word(self, device, float_val, multiplier=1000):
        self.write_word(device, int(round(float_val * multiplier)))

    # --- Heartbeat ----------------------------------------------------------
    def start_heartbeat(self, device, interval_sec=1.0):
        """Increment a 16-bit counter at `device` every `interval_sec` in a daemon thread."""
        if self._hb_thread and self._hb_thread.is_alive():
            return
        self._hb_stop.clear()

        def _run():
            counter = 0
            while not self._hb_stop.is_set():
                counter = (counter + 1) % 32760
                self.heartbeat_counter = counter
                self.write_word(device, counter)
                self._hb_stop.wait(interval_sec)

        self._hb_thread = threading.Thread(target=_run, name="plc-heartbeat", daemon=True)
        self._hb_thread.start()

    def stop_heartbeat(self):
        if self._hb_thread and self._hb_thread.is_alive():
            self._hb_stop.set()
            self._hb_thread.join(timeout=2.0)
        self._hb_thread = None

    # --- Slot helper --------------------------------------------------------
    def write_slot(self, slot_base_device, slot_index, words_per_slot, values):
        """Write one slot (consecutive words) at slot_base + slot_index*words_per_slot."""
        offset = slot_index * words_per_slot
        self.write_words(_offset_device(slot_base_device, offset), values)
