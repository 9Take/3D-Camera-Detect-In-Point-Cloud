"""Decode raw PLC words into a robot pose.

Pure integer/byte logic — no PLC connection needed, so it can be unit-tested
with hand-built word lists.
"""

POSE_SCALE = 1000.0   # PLC sends pose as int32 scaled x1000 (mm->um, deg->mdeg)


def decode_pose(words, swap=False):
    """Decode consecutive PLC words into scaled int32 values (/POSE_SCALE).

    Each value is two 16-bit words (low, high). swap=True => high-word-first.
    Returns a tuple of floats.
    """
    out = []
    for i in range(0, len(words) - 1, 2):
        lo, hi = words[i] & 0xFFFF, words[i + 1] & 0xFFFF
        if swap:
            lo, hi = hi, lo
        val = (hi << 16) | lo
        if val >= 0x80000000:          # sign-extend negative int32
            val -= 0x100000000
        out.append(val / POSE_SCALE)
    return tuple(out)
