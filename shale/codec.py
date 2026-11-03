"""
SHALE lossless audio codec (layer 1 of SHALE).

Frame-independent integer linear prediction (fixed polynomial predictors of
order 0-4 and quantised LPC of order 1..P) with partitioned Rice coding of the
residual and inter-channel (left/side, right/side, mid/side) decorrelation.
The design follows the well-known FLAC/Shorten family (RFC 9639); the bitstream
is SHALE's own and every frame can be decoded on its own.
"""
import numpy as np
from numba import njit

# --------------------------------------------------------------------------
# bit I/O (state: [byte position, accumulator, bits in accumulator])
# --------------------------------------------------------------------------


@njit(cache=True, inline="always")
def _put(buf, st, v, n):
    if n == 0:
        return
    acc = (st[1] << np.uint64(n)) | (np.uint64(v) & ((np.uint64(1) << np.uint64(n)) - np.uint64(1)))
    nb = st[2] + np.uint64(n)
    while nb >= np.uint64(8):
        nb -= np.uint64(8)
        buf[st[0]] = np.uint8((acc >> nb) & np.uint64(0xFF))
        st[0] += np.uint64(1)
    st[1] = acc & ((np.uint64(1) << nb) - np.uint64(1))
    st[2] = nb


@njit(cache=True)
def _flush(buf, st):
    if st[2] > np.uint64(0):
        _put(buf, st, 0, int(np.uint64(8) - st[2]))


@njit(cache=True, inline="always")
def _get(buf, st, n):
    # st: [bit position]
    v = np.uint64(0)
    for _ in range(n):
        p = st[0]
        bit = (np.uint64(buf[p >> np.uint64(3)]) >> (np.uint64(7) - (p & np.uint64(7)))) & np.uint64(1)
        v = (v << np.uint64(1)) | bit
        st[0] = p + np.uint64(1)
    return v


@njit(cache=True, inline="always")
def _get_fast(buf, st, n):
    """read n<=32 bits using byte steps"""
    if n == 0:
        return np.uint64(0)
    p = st[0]
    byte = p >> np.uint64(3)
    off = p & np.uint64(7)
    need = off + np.uint64(n)
    v = np.uint64(0)
    nbytes = (need + np.uint64(7)) >> np.uint64(3)
    for i in range(int(nbytes)):
        idx = byte + np.uint64(i)
        b = np.uint64(buf[idx]) if idx < np.uint64(buf.shape[0]) else np.uint64(0)
        v = (v << np.uint64(8)) | b
    v = (v >> (nbytes * np.uint64(8) - need)) & ((np.uint64(1) << np.uint64(n)) - np.uint64(1))
    st[0] = p + np.uint64(n)
    return v


@njit(cache=True, inline="always")
def _get_unary(buf, st):
    q = 0
    while True:
        p = st[0]
        bit = (buf[p >> np.uint64(3)] >> (7 - (p & np.uint64(7)))) & 1
        st[0] = p + np.uint64(1)
        if bit:
            return q
        q += 1


@njit(cache=True, inline="always")
def _signed(v, n):
    v = np.int64(v)
    if v >= (np.int64(1) << (n - 1)):
        v -= np.int64(1) << n
    return v


@njit(cache=True, inline="always")
def _zig(r):
    return np.uint64((r << 1) ^ (r >> 63))


@njit(cache=True, inline="always")
def _unzig(u):
    return np.int64(u >> np.uint64(1)) ^ -np.int64(u & np.uint64(1))

# --------------------------------------------------------------------------
# prediction
# --------------------------------------------------------------------------


@njit(cache=True)
def _fixed_residual(x, order, out):
    n = x.shape[0]
    for i in range(order):
        out[i] = 0
    for i in range(order, n):
        if order == 0:
            out[i] = x[i]
        elif order == 1:
            out[i] = x[i] - x[i - 1]
        elif order == 2:
            out[i] = x[i] - 2 * x[i - 1] + x[i - 2]
        elif order == 3:
            out[i] = x[i] - 3 * x[i - 1] + 3 * x[i - 2] - x[i - 3]
        else:
            out[i] = x[i] - 4 * x[i - 1] + 6 * x[i - 2] - 4 * x[i - 3] + x[i - 4]


@njit(cache=True)
def _lpc_residual(x, coef, order, shift, out):
    n = x.shape[0]
    for i in range(order):
        out[i] = 0
    for i in range(order, n):
        s = np.int64(0)
        for j in range(order):
            s += coef[j] * x[i - 1 - j]
        out[i] = x[i] - (s >> shift)


@njit(cache=True)
def _levinson(r, maxp, lpc):
    """lpc[p-1, :p] = predictor coefficients of order p (x[n] ~ sum a_j x[n-1-j])"""
    err = r[0]
    errs = np.zeros(maxp + 1)
    errs[0] = err
    a = np.zeros(maxp + 1)
    tmp = np.zeros(maxp + 1)
    for p in range(1, maxp + 1):
        if err <= 0:
            for q in range(p, maxp + 1):
                lpc[q - 1, :q] = 0.0
                errs[q] = 0.0
            return errs
        acc = r[p]
        for j in range(1, p):
            acc -= a[j] * r[p - j]
        k = acc / err
        tmp[:p] = a[:p]
        a[p] = k
        for j in range(1, p):
            a[j] = tmp[j] - k * tmp[p - j]
        err *= (1.0 - k * k)
        for j in range(p):
            lpc[p - 1, j] = a[j + 1]
        errs[p] = err
    return errs


@njit(cache=True)
def _quantise(a, order, prec, qc):
    cmax = 0.0
    for j in range(order):
        if abs(a[j]) > cmax:
            cmax = abs(a[j])
    if cmax <= 0.0:
        return -1
    e = int(np.floor(np.log2(cmax))) + 1      # cmax < 2^e
    shift = prec - 1 - e
    if shift > 20:
        shift = 20
    if shift < 0:
        return -1
    lim = (1 << (prec - 1)) - 1
    err = 0.0
    for j in range(order):
        err += a[j] * (1 << shift)
        q = int(np.round(err))
        if q > lim:
            q = lim
        if q < -lim - 1:
            q = -lim - 1
        qc[j] = q
        err -= q
    return shift

# --------------------------------------------------------------------------
# residual cost estimation and Rice coding
# --------------------------------------------------------------------------


@njit(cache=True)
def _rice_k(s, n):
    # optimal-ish Rice parameter for n values with sum s
    if n <= 0 or s <= 0:
        return 0
    m = s / n
    k = 0
    while k < 30 and (np.uint64(1) << np.uint64(k + 1)) <= m + 1:
        k += 1
    return k


@njit(cache=True)
def _best_partition(res, order, maxpo, n):
    """returns (estimated bits, partition order)"""
    # prefix sums of zigzag residuals
    ps = np.zeros(n + 1, np.uint64)
    for i in range(n):
        if i < order:
            ps[i + 1] = ps[i]
        else:
            ps[i + 1] = ps[i] + _zig(res[i])
    best = np.int64(1) << 62
    bpo = 0
    for po in range(maxpo + 1):
        np_ = 1 << po
        if n % np_ != 0:
            break
        plen = n // np_
        if plen <= order:
            break
        bits = np.int64(4)
        for p in range(np_):
            a = p * plen
            b = a + plen
            cnt = plen - order if p == 0 else plen
            s = ps[b] - ps[a]
            k = _rice_k(s, cnt)
            bits += 5 + cnt * (k + 1) + np.int64(s >> np.uint64(k))
        if bits < best:
            best = bits
            bpo = po
    return best, bpo


@njit(cache=True)
def _write_residual(buf, st, res, order, po, n):
    _put(buf, st, po, 4)
    np_ = 1 << po
    plen = n // np_
    for p in range(np_):
        a = p * plen + (order if p == 0 else 0)
        b = (p + 1) * plen
        s = np.uint64(0)
        mx = np.uint64(0)
        for i in range(a, b):
            u = _zig(res[i])
            s += u
            if u > mx:
                mx = u
        k = _rice_k(s, b - a)
        # refine k exactly among k-1, k, k+1
        bestk = k
        bestc = np.int64(1) << 62
        for kk in range(max(0, k - 1), min(30, k + 1) + 1):
            c = np.int64(0)
            for i in range(a, b):
                c += np.int64(_zig(res[i]) >> np.uint64(kk)) + kk + 1
            if c < bestc:
                bestc = c
                bestk = kk
        # escape if unary part could be huge
        w = 0
        while w < 40 and (np.uint64(1) << np.uint64(w)) <= mx:
            w += 1
        if np.int64(w) * (b - a) + 6 < bestc:
            _put(buf, st, 31, 5)
            _put(buf, st, w, 6)
            for i in range(a, b):
                _put(buf, st, _zig(res[i]), w)
        else:
            _put(buf, st, bestk, 5)
            kk = np.uint64(bestk)
            for i in range(a, b):
                u = _zig(res[i])
                q = u >> kk
                while q >= np.uint64(32):
                    _put(buf, st, 0, 32)
                    q -= np.uint64(32)
                _put(buf, st, 1, int(q) + 1)
                _put(buf, st, u & ((np.uint64(1) << kk) - np.uint64(1)), bestk)


@njit(cache=True)
def _read_residual(buf, st, res, order, n):
    po = int(_get_fast(buf, st, 4))
    np_ = 1 << po
    plen = n // np_
    for p in range(np_):
        a = p * plen + (order if p == 0 else 0)
        b = (p + 1) * plen
        k = int(_get_fast(buf, st, 5))
        if k == 31:
            w = int(_get_fast(buf, st, 6))
            for i in range(a, b):
                res[i] = _unzig(_get_fast(buf, st, w))
        else:
            kk = np.uint64(k)
            for i in range(a, b):
                q = np.uint64(_get_unary(buf, st))
                lo = _get_fast(buf, st, k)
                res[i] = _unzig((q << kk) | lo)

# --------------------------------------------------------------------------
# subframe encode / decode
# --------------------------------------------------------------------------
# types: 0 verbatim, 1 constant, 2 fixed, 3 lpc


@njit(cache=True)
def _encode_sub(buf, st, x, bits, maxp, prec, maxpo, res, best_res, qc, lpcmat, win, dry, exhaustive):
    n = x.shape[0]
    # constant?
    const = True
    for i in range(1, n):
        if x[i] != x[0]:
            const = False
            break
    if const:
        if not dry:
            _put(buf, st, 1, 3)
            _put(buf, st, np.uint64(x[0]) & ((np.uint64(1) << np.uint64(bits)) - np.uint64(1)), bits)
        return 3 + bits
    best = np.int64(3) + np.int64(bits) * n   # verbatim
    btype = 0
    border = 0
    bpo = 0
    bshift = 0
    # fixed predictors: pick order by sum |e| (one pass), then exact cost for the best
    e = np.zeros(5)
    for i in range(4, n):
        d0 = x[i]; d1 = d0 - x[i - 1]; d2 = d1 - (x[i - 1] - x[i - 2])
        d3 = d2 - (x[i - 1] - 2 * x[i - 2] + x[i - 3])
        d4 = d3 - (x[i - 1] - 3 * x[i - 2] + 3 * x[i - 3] - x[i - 4])
        e[0] += abs(d0); e[1] += abs(d1); e[2] += abs(d2); e[3] += abs(d3); e[4] += abs(d4)
    fo = 0
    for o in range(1, 5):
        if e[o] < e[fo]:
            fo = o
    if n <= 4 or exhaustive:
        olist = np.arange(0, min(5, n))
    else:
        olist = np.array([fo])
    for order in olist:
        _fixed_residual(x, order, res)
        c, po = _best_partition(res, order, maxpo, n)
        c += 3 + 3 + order * bits
        if c < best:
            best = c
            btype = 2
            border = order
            bpo = po
            best_res[:] = res[:]
    # LPC
    pmax = min(maxp, n - 1)
    if pmax >= 1:
        r = np.zeros(pmax + 1)
        xw = np.empty(n)
        for i in range(n):
            xw[i] = x[i] * win[i]
        for lag in range(pmax + 1):
            s = 0.0
            for i in range(lag, n):
                s += xw[i] * xw[i - lag]
            r[lag] = s
        if r[0] > 0:
            r[0] *= 1.0 + 1e-9
            errs = _levinson(r, pmax, lpcmat)
            if exhaustive:
                orders = np.arange(1, pmax + 1)
            else:
                # estimated bits per order from the Levinson prediction error
                be = 1e300
                bo = 1
                for o in range(1, pmax + 1):
                    ev = errs[o] / n
                    bps = 0.5 * np.log2(ev) if ev > 1e-12 else 0.0
                    if bps < 0:
                        bps = 0.0
                    est = n * bps + o * (prec + bits)
                    if est < be:
                        be = est
                        bo = o
                orders = np.array([bo])
            for order in orders:
                shift = _quantise(lpcmat[order - 1], order, prec, qc)
                if shift < 0:
                    continue
                _lpc_residual(x, qc, order, shift, res)
                c, po = _best_partition(res, order, maxpo, n)
                c += 3 + 5 + 4 + 5 + order * prec + order * bits
                if c < best:
                    best = c
                    btype = 3
                    border = order
                    bpo = po
                    bshift = shift
                    best_res[:] = res[:]
                    # remember coefficients
                    for j in range(order):
                        lpcmat[pmax, j] = qc[j]
    if dry:
        return best
    mask = (np.uint64(1) << np.uint64(bits)) - np.uint64(1)
    if btype == 0:
        _put(buf, st, 0, 3)
        for i in range(n):
            _put(buf, st, np.uint64(x[i]) & mask, bits)
    elif btype == 2:
        _put(buf, st, 2, 3)
        _put(buf, st, border, 3)
        for i in range(border):
            _put(buf, st, np.uint64(x[i]) & mask, bits)
        _write_residual(buf, st, best_res, border, bpo, n)
    else:
        _put(buf, st, 3, 3)
        _put(buf, st, border, 5)
        _put(buf, st, prec - 1, 4)
        _put(buf, st, bshift, 5)
        pm = (np.uint64(1) << np.uint64(prec)) - np.uint64(1)
        for j in range(border):
            _put(buf, st, np.uint64(np.int64(lpcmat[pmax, j])) & pm, prec)
        for i in range(border):
            _put(buf, st, np.uint64(x[i]) & mask, bits)
        _write_residual(buf, st, best_res, border, bpo, n)
    return best


@njit(cache=True)
def _decode_sub(buf, st, x, bits, res):
    n = x.shape[0]
    t = int(_get_fast(buf, st, 3))
    if t == 1:
        v = _signed(_get_fast(buf, st, bits), bits)
        for i in range(n):
            x[i] = v
    elif t == 0:
        for i in range(n):
            x[i] = _signed(_get_fast(buf, st, bits), bits)
    elif t == 2:
        order = int(_get_fast(buf, st, 3))
        for i in range(order):
            x[i] = _signed(_get_fast(buf, st, bits), bits)
        _read_residual(buf, st, res, order, n)
        for i in range(order, n):
            if order == 0:
                x[i] = res[i]
            elif order == 1:
                x[i] = res[i] + x[i - 1]
            elif order == 2:
                x[i] = res[i] + 2 * x[i - 1] - x[i - 2]
            elif order == 3:
                x[i] = res[i] + 3 * x[i - 1] - 3 * x[i - 2] + x[i - 3]
            else:
                x[i] = res[i] + 4 * x[i - 1] - 6 * x[i - 2] + 4 * x[i - 3] - x[i - 4]
    else:
        order = int(_get_fast(buf, st, 5))
        prec = int(_get_fast(buf, st, 4)) + 1
        shift = int(_get_fast(buf, st, 5))
        coef = np.empty(order, np.int64)
        for j in range(order):
            coef[j] = _signed(_get_fast(buf, st, prec), prec)
        for i in range(order):
            x[i] = _signed(_get_fast(buf, st, bits), bits)
        _read_residual(buf, st, res, order, n)
        for i in range(order, n):
            s = np.int64(0)
            for j in range(order):
                s += coef[j] * x[i - 1 - j]
            x[i] = res[i] + (s >> shift)

# --------------------------------------------------------------------------
# frame encode / decode
# --------------------------------------------------------------------------


@njit(cache=True)
def _tukey(n):
    w = np.ones(n)
    a = 0.5
    m = int(a * (n - 1) / 2)
    for i in range(m + 1):
        v = 0.5 * (1 - np.cos(2 * np.pi * i / (a * (n - 1)))) if n > 1 else 1.0
        w[i] = v
        w[n - 1 - i] = v
    return w


@njit(cache=True)
def encode_frame(frame, bits, maxp, prec, maxpo, out, exhaustive):
    """frame: (n, C) int64. Writes into out (uint8); returns number of bytes."""
    n, C = frame.shape
    st = np.zeros(3, np.uint64)
    res = np.zeros(n, np.int64)
    bres = np.zeros(n, np.int64)
    qc = np.zeros(33, np.int64)
    lpcmat = np.zeros((max(maxp, 1) + 1, 33))
    win = _tukey(n)
    if C == 2:
        L = frame[:, 0].copy()
        R = frame[:, 1].copy()
        S = L - R
        M = (L + R) >> 1
        cL = _encode_sub(out, st, L, bits, maxp, prec, maxpo, res, bres, qc, lpcmat, win, True, exhaustive)
        cR = _encode_sub(out, st, R, bits, maxp, prec, maxpo, res, bres, qc, lpcmat, win, True, exhaustive)
        cS = _encode_sub(out, st, S, bits + 1, maxp, prec, maxpo, res, bres, qc, lpcmat, win, True, exhaustive)
        cM = _encode_sub(out, st, M, bits, maxp, prec, maxpo, res, bres, qc, lpcmat, win, True, exhaustive)
        costs = np.array([cL + cR, cL + cS, cR + cS, cM + cS])
        mode = int(np.argmin(costs))
        _put(out, st, mode, 2)
        if mode == 0:
            _encode_sub(out, st, L, bits, maxp, prec, maxpo, res, bres, qc, lpcmat, win, False, exhaustive)
            _encode_sub(out, st, R, bits, maxp, prec, maxpo, res, bres, qc, lpcmat, win, False, exhaustive)
        elif mode == 1:
            _encode_sub(out, st, L, bits, maxp, prec, maxpo, res, bres, qc, lpcmat, win, False, exhaustive)
            _encode_sub(out, st, S, bits + 1, maxp, prec, maxpo, res, bres, qc, lpcmat, win, False, exhaustive)
        elif mode == 2:
            _encode_sub(out, st, R, bits, maxp, prec, maxpo, res, bres, qc, lpcmat, win, False, exhaustive)
            _encode_sub(out, st, S, bits + 1, maxp, prec, maxpo, res, bres, qc, lpcmat, win, False, exhaustive)
        else:
            _encode_sub(out, st, M, bits, maxp, prec, maxpo, res, bres, qc, lpcmat, win, False, exhaustive)
            _encode_sub(out, st, S, bits + 1, maxp, prec, maxpo, res, bres, qc, lpcmat, win, False, exhaustive)
    else:
        for c in range(C):
            x = frame[:, c].copy()
            _encode_sub(out, st, x, bits, maxp, prec, maxpo, res, bres, qc, lpcmat, win, False, exhaustive)
    _flush(out, st)
    return int(st[0])


@njit(cache=True)
def decode_frame(buf, n, C, bits, out):
    """out: (n, C) int64"""
    st = np.zeros(1, np.uint64)
    res = np.zeros(n, np.int64)
    if C == 2:
        mode = int(_get_fast(buf, st, 2))
        a = np.zeros(n, np.int64)
        b = np.zeros(n, np.int64)
        if mode == 0:
            _decode_sub(buf, st, a, bits, res)
            _decode_sub(buf, st, b, bits, res)
            out[:, 0] = a
            out[:, 1] = b
        elif mode == 1:
            _decode_sub(buf, st, a, bits, res)
            _decode_sub(buf, st, b, bits + 1, res)
            out[:, 0] = a
            out[:, 1] = a - b
        elif mode == 2:
            _decode_sub(buf, st, a, bits, res)
            _decode_sub(buf, st, b, bits + 1, res)
            out[:, 1] = a
            out[:, 0] = a + b
        else:
            _decode_sub(buf, st, a, bits, res)   # mid
            _decode_sub(buf, st, b, bits + 1, res)  # side
            for i in range(n):
                m = (a[i] << 1) | (b[i] & 1)
                out[i, 0] = (m + b[i]) >> 1
                out[i, 1] = (m - b[i]) >> 1
    else:
        for c in range(C):
            a = np.zeros(n, np.int64)
            _decode_sub(buf, st, a, bits, res)
            out[:, c] = a
    return int((st[0] + np.uint64(7)) >> np.uint64(3))


def max_frame_bytes(n, C, bits):
    # verbatim worst case + headers, side channel +1 bit
    return (n * C * (bits + 1) + 64 * C + 16) // 8 + 8


def encode_frames(x, bits, frame_len, maxp=12, prec=12, maxpo=8, exhaustive=False):
    """x: (N, C) integer array. Returns list of bytes objects, one per frame."""
    x = np.asarray(x, dtype=np.int64)
    if x.ndim == 1:
        x = x[:, None]
    N, C = x.shape
    out = np.zeros(max_frame_bytes(frame_len, C, bits), np.uint8)
    frames = []
    for s in range(0, N, frame_len):
        fr = np.ascontiguousarray(x[s:s + frame_len])
        nb = encode_frame(fr, bits, maxp, prec, maxpo, out, exhaustive)
        frames.append(out[:nb].tobytes())
    return frames


def decode_frames(frames, N, C, bits, frame_len):
    x = np.zeros((N, C), np.int64)
    for i, fb in enumerate(frames):
        s = i * frame_len
        n = min(frame_len, N - s)
        buf = np.frombuffer(fb + b"\x00" * 8, np.uint8)
        out = np.zeros((n, C), np.int64)
        decode_frame(buf, n, C, bits, out)
        x[s:s + n] = out
    return x
