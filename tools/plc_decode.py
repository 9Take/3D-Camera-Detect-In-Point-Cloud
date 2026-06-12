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


def int32_to_words(value, swap=False):
    """Split a 32-bit integer into two 16-bit words (low, high).

    Inverse of one decode step. Words are returned as signed int16 so a plain
    16-bit word-write stores the exact bit pattern decode_pose reads back.
    swap=True => high-word-first.
    """
    n = int(round(value))
    if n > 0x7FFFFFFF:   n = 0x7FFFFFFF      # clamp to int32 range
    if n < -0x80000000:  n = -0x80000000
    n &= 0xFFFFFFFF
    lo, hi = n & 0xFFFF, (n >> 16) & 0xFFFF
    if swap:
        lo, hi = hi, lo
    to_i16 = lambda w: w - 0x10000 if w >= 0x8000 else w
    return [to_i16(lo), to_i16(hi)]


def encode_pose(values, swap=False):
    """Inverse of decode_pose. `values` in mm/deg -> flat list of int16 words.

    Each value becomes two words (low, high) scaled by POSE_SCALE.
    """
    words = []
    for v in values:
        words += int32_to_words(v * POSE_SCALE, swap)
    return words
