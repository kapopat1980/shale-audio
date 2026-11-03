"""ctypes binding to the C lossless coder (same bitstream as codec.py)."""
import ctypes, os
import numpy as np
_lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), '..', 'lib', 'libshale_codec.so'))
_vp = ctypes.c_void_p
_lib.shale_encode_frames.argtypes = [_vp, ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, _vp, ctypes.c_int64, _vp]
_lib.shale_decode_frame.argtypes = [_vp, ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_int, _vp]
_P = lambda a: a.ctypes.data_as(_vp)


def max_frame_bytes(n, C, bits):
    return (n * C * (bits + 1) + 64 * C + 16) // 8 + 8


def encode_frames(x, bits, F, maxp=12, prec=12, maxpo=8):
    x = np.ascontiguousarray(np.asarray(x, np.int64).reshape(len(x), -1))
    N, C = x.shape
    nf = -(-N // F)
    mb = max_frame_bytes(F, C, bits)
    out = np.zeros(nf * mb, np.uint8)
    lens = np.zeros(nf, np.int64)
    _lib.shale_encode_frames(_P(x), N, C, bits, F, maxp, prec, maxpo, _P(out), mb, _P(lens))
    return [out[i * mb:i * mb + lens[i]].tobytes() for i in range(nf)]


def decode_frames(frames, N, C, bits, F):
    x = np.zeros((N, C), np.int64)
    for i, fb in enumerate(frames):
        n = min(F, N - i * F)
        buf = np.frombuffer(fb + bytes(8), np.uint8)
        o = np.zeros(n * C, np.int64)
        _lib.shale_decode_frame(_P(buf), len(fb), n, C, bits, _P(o))
        x[i * F:i * F + n] = o.reshape(n, C)
    return x


def decode_frame(buf, n, C, bits, out):
    """same signature as codec.decode_frame: buf uint8 array, out (n, C) int64"""
    o = np.zeros(n * C, np.int64)
    b = np.ascontiguousarray(buf, np.uint8)
    _lib.shale_decode_frame(_P(b), len(b), n, C, bits, _P(o))
    out[:] = o.reshape(n, C)
