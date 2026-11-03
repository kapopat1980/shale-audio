"""ctypes binding to SHALE's own Ascon implementation (csrc/ascon.c)."""
import ctypes, os
import numpy as np

_lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "..", "lib", "libshale_ascon.so"))
_u8p = ctypes.c_char_p
_sz = ctypes.c_size_t
_lib.ascon_aead128_encrypt.argtypes = [ctypes.c_void_p, ctypes.c_void_p, _sz, ctypes.c_void_p, _sz, ctypes.c_void_p, ctypes.c_void_p]
_lib.ascon_aead128_decrypt.argtypes = [ctypes.c_void_p, ctypes.c_void_p, _sz, ctypes.c_void_p, _sz, ctypes.c_void_p, ctypes.c_void_p]
_lib.ascon_xof128.argtypes = [ctypes.c_void_p, _sz, ctypes.c_void_p, _sz]
_lib.ascon_cxof128.argtypes = [ctypes.c_void_p, _sz, ctypes.c_void_p, _sz, ctypes.c_void_p, _sz]
_vp = ctypes.c_void_p
_lib.ascon_aead128_encrypt_batch.argtypes = [_vp, _vp, _vp, _vp, _vp, _vp, _sz, _vp, _vp, ctypes.c_int64]
_lib.ascon_aead128_decrypt_batch.argtypes = [_vp, _vp, _vp, _vp, _vp, _vp, _sz, _vp, _vp, ctypes.c_int64, _vp]


def _buf(b):
    return ctypes.create_string_buffer(bytes(b), max(1, len(b)))


def encrypt(key: bytes, nonce: bytes, ad: bytes, pt: bytes) -> bytes:
    assert len(key) == 16 and len(nonce) == 16
    out = ctypes.create_string_buffer(len(pt) + 16)
    _lib.ascon_aead128_encrypt(out, _buf(pt), len(pt), _buf(ad), len(ad), _buf(nonce), _buf(key))
    return out.raw


def decrypt(key: bytes, nonce: bytes, ad: bytes, ct: bytes):
    """Returns plaintext bytes, or None if the tag does not verify."""
    assert len(key) == 16 and len(nonce) == 16
    if len(ct) < 16:
        return None
    out = ctypes.create_string_buffer(max(1, len(ct) - 16))
    r = _lib.ascon_aead128_decrypt(out, _buf(ct), len(ct), _buf(ad), len(ad), _buf(nonce), _buf(key))
    return None if r != 0 else out.raw[: len(ct) - 16]


def xof(msg: bytes, outlen: int) -> bytes:
    out = ctypes.create_string_buffer(outlen)
    _lib.ascon_xof128(out, outlen, _buf(msg), len(msg))
    return out.raw


def cxof(msg: bytes, outlen: int, custom: bytes) -> bytes:
    out = ctypes.create_string_buffer(outlen)
    r = _lib.ascon_cxof128(out, outlen, _buf(msg), len(msg), _buf(custom), len(custom))
    assert r == 0
    return out.raw


def _ptr(a):
    return a.ctypes.data_as(ctypes.c_void_p)


def encrypt_batch(key, nonces, ads, m, moff, mlen):
    """nonces: (nf,16) uint8, ads: (nf,A) uint8, m: uint8 buffer, moff/mlen: int64.
    Returns (ciphertext buffer, coff)."""
    nf = len(mlen)
    clen = mlen + 16
    coff = np.zeros(nf, np.int64)
    if nf > 1:
        coff[1:] = np.cumsum(clen)[:-1]
    c = np.zeros(int(clen.sum()), np.uint8)
    k = np.frombuffer(key, np.uint8).copy()
    _lib.ascon_aead128_encrypt_batch(_ptr(c), _ptr(coff), _ptr(m), _ptr(moff), _ptr(mlen),
                                     _ptr(ads), ads.shape[1], _ptr(nonces), _ptr(k), nf)
    return c, coff


def decrypt_batch(key, nonces, ads, c, coff, clen):
    nf = len(clen)
    mlen = clen - 16
    moff = np.zeros(nf, np.int64)
    if nf > 1:
        moff[1:] = np.cumsum(mlen)[:-1]
    m = np.zeros(max(1, int(mlen.sum())), np.uint8)
    ok = np.zeros(nf, np.int32)
    k = np.frombuffer(key, np.uint8).copy()
    _lib.ascon_aead128_decrypt_batch(_ptr(m), _ptr(moff), _ptr(c), _ptr(coff), _ptr(clen),
                                     _ptr(ads), ads.shape[1], _ptr(nonces), _ptr(k), nf, _ptr(ok))
    return m, moff, ok.astype(bool)
