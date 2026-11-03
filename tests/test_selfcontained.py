"""Tests that need no external data: bit-exact round trips (Numba and C coders), identical
bitstreams from both coders, stream/file/SIV modes, and tamper rejection."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from shale import core, eclq, codec, ccodec

rng = np.random.default_rng(1)
K = os.urandom(16)
t = np.arange(48000) / 16000


def signals():
    speechlike = (8000 * np.sin(2 * np.pi * 180 * t) * (1 + np.sin(2 * np.pi * 3 * t)) + rng.normal(0, 200, len(t))).astype(np.int64)
    silence = np.zeros(20000, np.int64)
    noise = rng.integers(-32768, 32768, 30000)
    stereo = np.stack([speechlike, (0.6 * speechlike + rng.normal(0, 300, len(t))).astype(np.int64)], 1)
    s24 = (rng.normal(0, 2 ** 18, (15000, 2))).astype(np.int64)
    return [(speechlike[:, None], 16), (silence[:, None], 16), (noise[:, None], 16), (stereo, 16), (s24, 24)]


def main():
    fails = 0
    for x, bits in signals():
        N, C = x.shape
        for F in (320, 4096):
            a = codec.encode_frames(x, bits, F); b = ccodec.encode_frames(x, bits, F)
            fails += a != b
            fails += not np.array_equal(codec.decode_frames(a, N, C, bits, F), x)
            fails += not np.array_equal(ccodec.decode_frames(b, N, C, bits, F), x)
        top = core.frame_top(320, C, bits)
        h, recs, idx = core.encrypt_stream(K, x, 16000, bits, 320, eclq.uniform_buckets(top, 4))
        fails += not np.array_equal(core.decrypt_stream(K, h, recs), x)
        blob, _ = core.encrypt_file(K, x, 16000, bits, 4096, [0.5, 0.7, 1.0])
        fails += not np.array_equal(core.decrypt_file(K, blob, [0.5, 0.7, 1.0]), x)
        sblob, _ = core.encrypt_file_siv(K, x, 16000, bits, 4096, [0.5, 0.7, 1.0])
        fails += not np.array_equal(core.decrypt_file_siv(K, sblob, [0.5, 0.7, 1.0]), x)
        for bl, dec in ((blob, core.decrypt_file), (sblob, core.decrypt_file_siv)):
            bb = bytearray(bl); bb[len(bb) // 2] ^= 1
            try:
                dec(K, bytes(bb), [0.5, 0.7, 1.0]); fails += 1
            except core.AuthError:
                pass
    print('self-contained tests:', 'PASS' if fails == 0 else f'{fails} FAILURES')
    return fails


if __name__ == '__main__':
    sys.exit(1 if main() else 0)
