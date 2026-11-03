"""
SHALE: Size-Hiding Authenticated Lossless audio Encryption.

  layer 1  lossless codec (codec.py)            -> variable-length frame payloads
  layer 2  ECLQ length quantiser (eclq.py)      -> every observable size is a bucket
  layer 3  Ascon-AEAD128 (NIST SP 800-232)       -> confidentiality + integrity
           keys from Ascon-CXOF128 over (master key, random 128-bit salt)

Two modes:
  stream : one AEAD record per frame, padded to a frame bucket (live audio / VoIP)
  file   : whole payload padded to a file-level bucket, cut into fixed 64 KiB chunks
"""
import os, struct
import numpy as np
from . import codec, eclq, ascon

MAGIC = b"SHL1"
CHUNK = 65536
VERB = 0x80000000          # verbatim flag in the 32-bit record length field


class AuthError(Exception):
    pass


# ---------------------------------------------------------------- AEAD backends
class _Ascon:
    name = "Ascon-AEAD128"
    keylen = 16

    def __init__(self, key):
        self.k = key

    def enc(self, i, final, ad, pt):
        return ascon.encrypt(self.k, struct.pack("<QQ", i, final), ad, pt)

    def dec(self, i, final, ad, ct):
        return ascon.decrypt(self.k, struct.pack("<QQ", i, final), ad, ct)


class _AesGcm:
    name = "AES-128-GCM"
    keylen = 16

    def __init__(self, key):
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        self.a = AESGCM(key)

    def enc(self, i, final, ad, pt):
        return self.a.encrypt(struct.pack("<QI", i, final), pt, ad)

    def dec(self, i, final, ad, ct):
        from cryptography.exceptions import InvalidTag
        try:
            return self.a.decrypt(struct.pack("<QI", i, final), ct, ad)
        except InvalidTag:
            return None


AEADS = {"ascon": _Ascon, "aesgcm": _AesGcm}


def derive_key(master, salt, aead="ascon"):
    return ascon.cxof(master + salt, AEADS[aead].keylen, b"SHALE/v1/record-key")


def header_digest(h):
    return ascon.cxof(h, 16, b"SHALE/v1/header")


# ---------------------------------------------------------------- PCM helpers
def pcm_bytes(x, bits):
    x = np.asarray(x, np.int64).reshape(-1)
    if bits == 16:
        return x.astype("<i2").tobytes()
    b = x.astype("<i4").view(np.uint8).reshape(-1, 4)[:, :3]
    return b.tobytes()


def pcm_from_bytes(buf, bits, n, C):
    if bits == 16:
        return np.frombuffer(buf, "<i2", count=n * C).astype(np.int64).reshape(n, C)
    a = np.frombuffer(buf, np.uint8, count=3 * n * C).reshape(-1, 3).astype(np.int64)
    v = a[:, 0] | (a[:, 1] << 8) | (a[:, 2] << 16)
    v = np.where(v >= 1 << 23, v - (1 << 24), v)
    return v.reshape(n, C)


USE_C = False      # set True to use the C coder (identical bitstream, faster)


def _decode_frame(buf, n, C, bits, out):
    if USE_C:
        from . import ccodec
        return ccodec.decode_frame(buf, n, C, bits, out)
    return codec.decode_frame(buf, n, C, bits, out)


def frame_payloads(x, bits, F, **kw):
    """codec frames with verbatim fallback; returns list of (flag|len, bytes) records"""
    x = np.asarray(x, np.int64)
    if x.ndim == 1:
        x = x[:, None]
    if USE_C:
        from . import ccodec
        frames = ccodec.encode_frames(x, bits, F, **kw)
    else:
        frames = codec.encode_frames(x, bits, F, **kw)
    out = []
    for i, fb in enumerate(frames):
        raw = pcm_bytes(x[i * F:(i + 1) * F], bits)
        if len(fb) >= len(raw):
            out.append((VERB | len(raw), raw))
        else:
            out.append((len(fb), fb))
    return out


def raw_frame_bytes(F, C, bits):
    return F * C * bits // 8


def frame_top(F, C, bits):
    """largest possible record plaintext: 4-byte length field + verbatim frame"""
    return raw_frame_bytes(F, C, bits) + 4


# ---------------------------------------------------------------- header
def make_header(mode, fs, C, bits, N, F, aead, buckets, salt):
    bdig = ascon.xof(np.asarray(buckets, "<u8").tobytes(), 8)
    return (MAGIC + struct.pack("<BBIBBQIB", 1, 0 if mode == "stream" else 1, fs, C, bits, N, F,
                                0 if aead == "ascon" else 1) + bdig + salt)


def parse_header(h):
    assert h[:4] == MAGIC
    v, m, fs, C, bits, N, F, a = struct.unpack("<BBIBBQIB", h[4:25])
    return dict(mode="stream" if m == 0 else "file", fs=fs, C=C, bits=bits, N=N, F=F,
                aead="ascon" if a == 0 else "aesgcm", bdig=h[25:33], salt=h[33:49])


HDR_LEN = 49


# ---------------------------------------------------------------- stream mode
def encrypt_stream(master, x, fs, bits, F, buckets, aead="ascon", salt=None, payloads=None):
    """Returns (header, list of ciphertext records, list of bucket indices)."""
    x = np.asarray(x, np.int64)
    if x.ndim == 1:
        x = x[:, None]
    N, C = x.shape
    buckets = np.asarray(buckets, np.int64)
    assert buckets[-1] >= frame_top(F, C, bits)
    salt = os.urandom(16) if salt is None else salt
    h = make_header("stream", fs, C, bits, N, F, aead, buckets, salt)
    hd = header_digest(h)
    E = AEADS[aead](derive_key(master, salt, aead))
    if payloads is None:
        payloads = frame_payloads(x, bits, F)
    recs, idxs = [], []
    n = len(payloads)
    for i, (lf, body) in enumerate(payloads):
        L = 4 + len(body)
        j = int(np.searchsorted(buckets, L, side="left"))
        pt = struct.pack("<I", lf) + body + bytes(int(buckets[j]) - L)
        fin = 1 if i == n - 1 else 0
        recs.append(E.enc(i, fin, hd + struct.pack("<QB", i, fin), pt))
        idxs.append(j)
    return h, recs, idxs


def decrypt_stream(master, h, recs):
    p = parse_header(h)
    N, C, bits, F = p["N"], p["C"], p["bits"], p["F"]
    hd = header_digest(h)
    E = AEADS[p["aead"]](derive_key(master, p["salt"], p["aead"]))
    nexp = -(-N // F)
    if len(recs) != nexp:
        raise AuthError("record count does not match header (truncation or extension)")
    x = np.zeros((N, C), np.int64)
    for i, ct in enumerate(recs):
        fin = 1 if i == nexp - 1 else 0
        pt = E.dec(i, fin, hd + struct.pack("<QB", i, fin), ct)
        if pt is None:
            raise AuthError(f"record {i} failed authentication")
        lf = struct.unpack("<I", pt[:4])[0]
        n = min(F, N - i * F)
        if lf & VERB:
            x[i * F:i * F + n] = pcm_from_bytes(pt[4:4 + (lf & ~VERB)], bits, n, C)
        else:
            buf = np.frombuffer(pt[4:4 + lf] + bytes(8), np.uint8)
            out = np.zeros((n, C), np.int64)
            _decode_frame(buf, n, C, bits, out)
            x[i * F:i * F + n] = out
    return x


# ---------------------------------------------------------------- file mode
def file_top(N, C, bits, F):
    """largest possible file plaintext: 8-byte total + 4-byte length per frame + verbatim audio"""
    n = -(-N // F)
    return 8 + 4 * n + N * C * bits // 8


def file_padded_size(T, N, C, bits, F, ratios):
    """ratios: ascending bucket ratios relative to file_top (the verbatim worst case), last = 1.0.
    Returns padded plaintext size and bucket index."""
    top = file_top(N, C, bits, F)
    sizes = [int(np.ceil(r * top)) for r in ratios[:-1]] + [top]
    for j, s in enumerate(sizes):
        if T <= s:
            return s, j
    return top, len(sizes) - 1


def encrypt_file(master, x, fs, bits, F, ratios, aead="ascon", salt=None, payloads=None):
    """Returns ciphertext bytes (header || chunks) and the file bucket index."""
    x = np.asarray(x, np.int64)
    if x.ndim == 1:
        x = x[:, None]
    N, C = x.shape
    salt = os.urandom(16) if salt is None else salt
    h = make_header("file", fs, C, bits, N, F, aead, np.round(np.asarray(ratios) * 1e6).astype(np.int64), salt)
    hd = header_digest(h)
    E = AEADS[aead](derive_key(master, salt, aead))
    if payloads is None:
        payloads = frame_payloads(x, bits, F)
    table = b"".join(struct.pack("<I", lf) for lf, _ in payloads)
    body = table + b"".join(b for _, b in payloads)
    T = 8 + len(body)
    P, j = file_padded_size(T, N, C, bits, F, ratios)
    pt = struct.pack("<Q", T) + body + bytes(P - T)
    nch = -(-len(pt) // CHUNK)
    out = [h]
    for c in range(nch):
        fin = 1 if c == nch - 1 else 0
        out.append(E.enc(c, fin, hd + struct.pack("<QB", c, fin), pt[c * CHUNK:(c + 1) * CHUNK]))
    return b"".join(out), j


def decrypt_file(master, blob, ratios):
    h = blob[:HDR_LEN]
    p = parse_header(h)
    N, C, bits, F = p["N"], p["C"], p["bits"], p["F"]
    hd = header_digest(h)
    E = AEADS[p["aead"]](derive_key(master, p["salt"], p["aead"]))
    body = blob[HDR_LEN:]
    # the padded size can only be one of the bucket sizes; recover chunk layout from length
    nch = -(-len(body) // (CHUNK + 16))
    pt = []
    for c in range(nch):
        ct = body[c * (CHUNK + 16):(c + 1) * (CHUNK + 16)]
        fin = 1 if c == nch - 1 else 0
        m = E.dec(c, fin, hd + struct.pack("<QB", c, fin), ct)
        if m is None:
            raise AuthError(f"chunk {c} failed authentication")
        pt.append(m)
    pt = b"".join(pt)
    T = struct.unpack("<Q", pt[:8])[0]
    n = -(-N // F)
    lens = struct.unpack(f"<{n}I", pt[8:8 + 4 * n])
    pos = 8 + 4 * n
    x = np.zeros((N, C), np.int64)
    for i, lf in enumerate(lens):
        m = min(F, N - i * F)
        if lf & VERB:
            L = lf & ~VERB
            x[i * F:i * F + m] = pcm_from_bytes(pt[pos:pos + L], bits, m, C)
        else:
            L = lf
            buf = np.frombuffer(pt[pos:pos + L] + bytes(8), np.uint8)
            out = np.zeros((m, C), np.int64)
            _decode_frame(buf, m, C, bits, out)
            x[i * F:i * F + m] = out
        pos += L
    if pos != T:
        raise AuthError("inconsistent payload length")
    return x


# ---------------------------------------------------------------- SIV file mode (nonce-misuse resistant)
def encrypt_file_siv(master, x, fs, bits, F, ratios, salt=None, payloads=None):
    """Deterministic, misuse-resistant variant of file mode.
    SIV = Ascon-CXOF128(K_mac || header || padded plaintext); records are then encrypted
    under K_enc = Ascon-CXOF128(K || SIV) with counter nonces. Reusing (or omitting) the
    salt reveals only whether two files are identical."""
    x = np.asarray(x, np.int64)
    if x.ndim == 1:
        x = x[:, None]
    N, C = x.shape
    salt = bytes(16) if salt is None else salt
    h = make_header("file", fs, C, bits, N, F, "ascon", np.round(np.asarray(ratios) * 1e6).astype(np.int64), salt)
    h = h[:4] + bytes([2]) + h[5:]                      # version byte 2 = SIV
    if payloads is None:
        payloads = frame_payloads(x, bits, F)
    table = b"".join(struct.pack("<I", lf) for lf, _ in payloads)
    body = table + b"".join(b for _, b in payloads)
    T = 8 + len(body)
    P, j = file_padded_size(T, N, C, bits, F, ratios)
    pt = struct.pack("<Q", T) + body + bytes(P - T)
    kmac = ascon.cxof(master + salt, 16, b"SHALE/v1/siv-mac-key")
    siv = ascon.cxof(kmac + h + pt, 16, b"SHALE/v1/siv")
    E = _Ascon(ascon.cxof(master + siv, 16, b"SHALE/v1/siv-enc-key"))
    hd = header_digest(h)
    nch = -(-len(pt) // CHUNK)
    out = [h, siv]
    for c in range(nch):
        fin = 1 if c == nch - 1 else 0
        out.append(E.enc(c, fin, hd + struct.pack("<QB", c, fin), pt[c * CHUNK:(c + 1) * CHUNK]))
    return b"".join(out), j


def decrypt_file_siv(master, blob, ratios):
    h = blob[:HDR_LEN]
    siv = blob[HDR_LEN:HDR_LEN + 16]
    p = parse_header(h)
    N, C, bits, F = p["N"], p["C"], p["bits"], p["F"]
    hd = header_digest(h)
    E = _Ascon(ascon.cxof(master + siv, 16, b"SHALE/v1/siv-enc-key"))
    body = blob[HDR_LEN + 16:]
    nch = -(-len(body) // (CHUNK + 16))
    pt = []
    for c in range(nch):
        fin = 1 if c == nch - 1 else 0
        m = E.dec(c, fin, hd + struct.pack("<QB", c, fin), body[c * (CHUNK + 16):(c + 1) * (CHUNK + 16)])
        if m is None:
            raise AuthError(f"chunk {c} failed authentication")
        pt.append(m)
    pt = b"".join(pt)
    kmac = ascon.cxof(master + p["salt"], 16, b"SHALE/v1/siv-mac-key")
    if ascon.cxof(kmac + h + pt, 16, b"SHALE/v1/siv") != siv:
        raise AuthError("synthetic IV mismatch")
    # reuse the normal parser on a reconstructed nonce-mode layout
    T = struct.unpack("<Q", pt[:8])[0]
    n = -(-N // F)
    lens = struct.unpack(f"<{n}I", pt[8:8 + 4 * n])
    pos = 8 + 4 * n
    x = np.zeros((N, C), np.int64)
    for i, lf in enumerate(lens):
        m = min(F, N - i * F)
        L = lf & ~VERB
        if lf & VERB:
            x[i * F:i * F + m] = pcm_from_bytes(pt[pos:pos + L], bits, m, C)
        else:
            out = np.zeros((m, C), np.int64)
            _decode_frame(np.frombuffer(pt[pos:pos + L] + bytes(8), np.uint8), m, C, bits, out)
            x[i * F:i * F + m] = out
        pos += L
    if pos != T:
        raise AuthError("inconsistent payload length")
    return x
