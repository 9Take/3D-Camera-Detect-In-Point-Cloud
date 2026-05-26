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
        self._last_reconnect_attempt = 0.0
        self._reconnect_cooldown_sec = 2.0

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
            try: self.plc.close()
            except Exception: pass
            self.connected = False
            print("[PLC] Disconnected.")

    # --- Internal: reconnect + retry wrapper --------------------------------
    def _try_reconnect(self):
        """Attempt to reopen the MC protocol session. Rate-limited to avoid hammering."""
        now = time.time()
        if now - self._last_reconnect_attempt < self._reconnect_cooldown_sec:
            return False
        self._last_reconnect_attempt = now
        try: self.plc.close()
        except Exception: pass
        try:
            self.plc.connect(self.ip, self.port)
            self.connected = True
            print("[PLC] Reconnected.")
            return True
        except Exception as e:
            print(f"[PLC ERROR] Reconnect failed: {e}")
            self.connected = False
            return False

    def _call(self, label, fn):
        """Run `fn()` under the lock with one reconnect-and-retry on failure.
        Returns (success, result_or_None)."""
        if not self.connected:
            with self._lock:
                if not self._try_reconnect():
                    return False, None
        with self._lock:
            try:
                return True, fn()
            except Exception as e:
                print(f"[PLC ERROR] {label} failed: {e} — attempting reconnect.")
                self.connected = False
                if not self._try_reconnect():
                    return False, None
                try:
                    return True, fn()
                except Exception as e2:
                    print(f"[PLC ERROR] {label} failed after reconnect: {e2}")
                    self.connected = False
                    return False, None

    # --- Reads (return safe default on failure; signature unchanged) --------
    def read_bit(self, device):
        ok, res = self._call(f"Read Bit {device}",
                             lambda: self.plc.batchread_bitunits(device, 1))
        return res if ok else [0]

    def read_word(self, device):
        ok, res = self._call(f"Read Word {device}",
                             lambda: self.plc.batchread_wordunits(device, 1)[0])
        return res if ok else 0

    def read_bits(self, device, count):
        if count <= 0: return []
        ok, res = self._call(f"Read Bits {device} (n={count})",
                             lambda: self.plc.batchread_bitunits(device, count))
        return res if ok else [0] * count

    def read_words(self, device, count):
        if count <= 0: return []
        ok, res = self._call(f"Read Words {device} (n={count})",
                             lambda: self.plc.batchread_wordunits(device, count))
        return res if ok else [0] * count

    # --- Writes (return True on success, False on failure) ------------------
    def write_bit(self, device, value):
        ok, _ = self._call(f"Write Bit {device}",
                           lambda: self.plc.batchwrite_bitunits(device, [int(value)]))
        return ok

    def write_word(self, device, value):
        ok, _ = self._call(f"Write Word {device}",
                           lambda: self.plc.batchwrite_wordunits(device, [_clamp_int16(value)]))
        return ok

    def write_bits(self, device, values):
        if not values: return True
        clamped = [int(bool(v)) for v in values]
        ok, _ = self._call(f"Write Bits {device} (n={len(values)})",
                           lambda: self.plc.batchwrite_bitunits(device, clamped))
        return ok

    def write_words(self, device, values):
        if not values: return True
        clamped = [_clamp_int16(v) for v in values]
        ok, _ = self._call(f"Write Words {device} (n={len(values)})",
                           lambda: self.plc.batchwrite_wordunits(device, clamped))
        return ok

    def write_scaled_word(self, device, float_val, multiplier=1000):
        return self.write_word(device, int(round(float_val * multiplier)))

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
        return self.write_words(_offset_device(slot_base_device, offset), values)
