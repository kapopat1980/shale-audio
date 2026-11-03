"""In-house implementation of NIST SP 800-22 rev.1a tests (subset), following the formulas
of the specification. Validated on os.urandom before use."""
import numpy as np
from scipy.special import gammaincc, erfc
from scipy.stats import norm
from numba import njit


def monobit(b):
    n = len(b)
    s = np.abs(np.sum(2 * b.astype(np.int64) - 1)) / np.sqrt(n)
    return erfc(s / np.sqrt(2))


def block_frequency(b, M=128):
    n = len(b) // M
    pi = b[:n * M].reshape(n, M).mean(1)
    chi = 4 * M * np.sum((pi - 0.5) ** 2)
    return gammaincc(n / 2, chi / 2)


def runs(b):
    n = len(b)
    pi = b.mean()
    if abs(pi - 0.5) >= 2 / np.sqrt(n):
        return 0.0
    v = 1 + np.count_nonzero(b[1:] != b[:-1])
    return erfc(abs(v - 2 * n * pi * (1 - pi)) / (2 * np.sqrt(2 * n) * pi * (1 - pi)))


@njit(cache=True)
def _longest_runs(b, M):
    n = len(b) // M
    out = np.zeros(n, np.int64)
    for i in range(n):
        best = 0
        cur = 0
        for j in range(M):
            if b[i * M + j]:
                cur += 1
                if cur > best:
                    best = cur
            else:
                cur = 0
        out[i] = best
    return out


def longest_run(b):
    # n >= 750000: M = 10^4, K = 6, classes <=10, 11..15, >=16
    M = 10000
    v = np.array([10, 11, 12, 13, 14, 15, 16])
    pi = np.array([0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727])
    L = _longest_runs(b.astype(np.uint8), M)
    N = len(L)
    cnt = np.array([np.sum(L <= 10)] + [np.sum(L == k) for k in range(11, 16)] + [np.sum(L >= 16)])
    chi = np.sum((cnt - N * pi) ** 2 / (N * pi))
    return gammaincc(6 / 2, chi / 2)


@njit(cache=True)
def _rank_gf2(m):
    m = m.copy()
    rows, cols = m.shape
    r = 0
    for c in range(cols):
        piv = -1
        for i in range(r, rows):
            if m[i, c]:
                piv = i
                break
        if piv < 0:
            continue
        if piv != r:
            for k in range(cols):
                t = m[r, k]; m[r, k] = m[piv, k]; m[piv, k] = t
        for i in range(rows):
            if i != r and m[i, c]:
                for k in range(cols):
                    m[i, k] ^= m[r, k]
        r += 1
        if r == rows:
            break
    return r


def matrix_rank(b):
    Q = 32
    N = len(b) // (Q * Q)
    ranks = np.array([_rank_gf2(b[i * Q * Q:(i + 1) * Q * Q].reshape(Q, Q).astype(np.uint8)) for i in range(N)])
    F32 = np.sum(ranks == 32); F31 = np.sum(ranks == 31); rest = N - F32 - F31
    p = np.array([0.2888, 0.5776, 0.1336])
    chi = (F32 - p[0] * N) ** 2 / (p[0] * N) + (F31 - p[1] * N) ** 2 / (p[1] * N) + (rest - p[2] * N) ** 2 / (p[2] * N)
    return np.exp(-chi / 2)


def dft(b):
    n = len(b)
    x = 2 * b.astype(float) - 1
    S = np.abs(np.fft.rfft(x))[: n // 2]
    T = np.sqrt(np.log(1 / 0.05) * n)
    N0 = 0.95 * n / 2
    N1 = np.sum(S < T)
    d = (N1 - N0) / np.sqrt(n * 0.95 * 0.05 / 4)
    return erfc(abs(d) / np.sqrt(2))


def universal(b, L=7, Q=1280):
    n = len(b)
    K = n // L - Q
    w = b[: (Q + K) * L].reshape(-1, L).astype(np.int64) @ (1 << np.arange(L - 1, -1, -1))
    last = np.zeros(1 << L, np.int64)
    for i in range(Q):
        last[w[i]] = i + 1
    s = _universal_sum(w, last, Q, K)
    fn = s / K
    ev, var = 6.1962507, 3.125
    c = 0.7 - 0.8 / L + (4 + 32 / L) * K ** (-3 / L) / 15
    sigma = c * np.sqrt(var / K)
    return erfc(abs(fn - ev) / (np.sqrt(2) * sigma))


@njit(cache=True)
def _universal_sum(w, last, Q, K):
    s = 0.0
    for i in range(Q, Q + K):
        s += np.log2(i + 1 - last[w[i]])
        last[w[i]] = i + 1
    return s


def _psi2(b, m):
    if m <= 0:
        return 0.0
    n = len(b)
    ext = np.concatenate([b, b[: m - 1]]).astype(np.int64)
    idx = np.zeros(n, np.int64)
    for k in range(m):
        idx = (idx << 1) | ext[k:k + n]
    c = np.bincount(idx, minlength=1 << m)
    return (1 << m) / n * np.sum(c.astype(float) ** 2) - n


def serial(b, m=16):
    p0, p1, p2 = _psi2(b, m), _psi2(b, m - 1), _psi2(b, m - 2)
    d1 = p0 - p1
    d2 = p0 - 2 * p1 + p2
    return gammaincc(2 ** (m - 2), d1 / 2), gammaincc(2 ** (m - 3), d2 / 2)


def _phi(b, m):
    n = len(b)
    ext = np.concatenate([b, b[: m - 1]]).astype(np.int64)
    idx = np.zeros(n, np.int64)
    for k in range(m):
        idx = (idx << 1) | ext[k:k + n]
    c = np.bincount(idx, minlength=1 << m) / n
    c = c[c > 0]
    return np.sum(c * np.log(c))


def approximate_entropy(b, m=10):
    n = len(b)
    ap = _phi(b, m) - _phi(b, m + 1)
    chi = 2 * n * (np.log(2) - ap)
    return gammaincc(2 ** (m - 1), chi / 2)


def cusum(b, reverse=False):
    x = 2 * b.astype(np.int64) - 1
    if reverse:
        x = x[::-1]
    z = np.max(np.abs(np.cumsum(x)))
    n = len(b)
    k1 = np.arange(int((-n / z + 1) / 4), int((n / z - 1) / 4) + 1)
    k2 = np.arange(int((-n / z - 3) / 4), int((n / z - 1) / 4) + 1)
    s1 = np.sum(norm.cdf((4 * k1 + 1) * z / np.sqrt(n)) - norm.cdf((4 * k1 - 1) * z / np.sqrt(n)))
    s2 = np.sum(norm.cdf((4 * k2 + 3) * z / np.sqrt(n)) - norm.cdf((4 * k2 + 1) * z / np.sqrt(n)))
    return 1 - s1 + s2


TESTS = ['Frequency (monobit)', 'Block frequency (M=128)', 'Runs', 'Longest run of ones (M=10^4)',
         'Binary matrix rank (32x32)', 'Discrete Fourier transform', "Maurer's universal (L=7)",
         'Approximate entropy (m=10)', 'Serial 1 (m=16)', 'Serial 2 (m=16)',
         'Cumulative sums (forward)', 'Cumulative sums (backward)']


def battery(b):
    s1, s2 = serial(b)
    return [monobit(b), block_frequency(b), runs(b), longest_run(b), matrix_rank(b), dft(b),
            universal(b), approximate_entropy(b), s1, s2, cusum(b), cusum(b, True)]


def summarize(P, alpha=0.01):
    """P: (sequences, tests) p-values -> pass proportion and uniformity P-value per test"""
    from scipy.stats import chisquare
    n = P.shape[0]
    out = []
    for j in range(P.shape[1]):
        p = P[:, j]
        prop = np.mean(p >= alpha)
        hist = np.histogram(p, bins=10, range=(0, 1))[0]
        pt = gammaincc(9 / 2, chisquare(hist)[0] / 2)
        out.append((prop, pt))
    lo = (1 - alpha) - 3 * np.sqrt(alpha * (1 - alpha) / n)
    return out, lo
