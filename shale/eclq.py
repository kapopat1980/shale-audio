"""
Entropy-Constrained Length Quantisation (ECLQ) - SHALE layer 2.

Given the public distribution of compressed record lengths for one audio profile,
choose bucket sizes b_1 < ... < b_K (b_K = the uncompressed worst case) that minimise

        E[b(l) - l]  +  lambda * H(B)        subject to  K <= Kmax,

where b(l) is the smallest bucket >= l and B is the bucket index. The objective
decomposes over contiguous buckets, so an exact dynamic program over a grid of
candidate bucket tops finds the optimum. lambda = 0 gives the minimum-overhead
K-bucket quantiser; a bisection on lambda meets a target entropy (leakage) budget.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def _dp(tops, mass, lsum, kmax, lam):
    """tops: (M,) candidate tops (ascending, last = forced top).
    mass[m], lsum[m]: probability mass and sum of l*p(l) of lengths in (tops[m-1], tops[m]].
    Returns choice table for reconstruction: best cost per k, and back pointers."""
    M = tops.shape[0]
    W = np.zeros(M + 1)
    S = np.zeros(M + 1)
    for m in range(M):
        W[m + 1] = W[m] + mass[m]
        S[m + 1] = S[m] + lsum[m]
    INF = 1e300
    D = np.full((kmax + 1, M + 1), INF)
    back = np.full((kmax + 1, M + 1), -1, np.int64)
    D[0, 0] = 0.0
    for k in range(1, kmax + 1):
        for j in range(1, M + 1):
            # bucket covers cells i..j-1 with top tops[j-1]
            # skip empty top cells except the forced last one (no gain from empty buckets)
            if mass[j - 1] <= 0.0 and j != M:
                continue
            best = INF
            bi = -1
            for i in range(0, j):
                if D[k - 1, i] >= INF:
                    continue
                q = W[j] - W[i]
                pad = q * tops[j - 1] - (S[j] - S[i])
                ent = -q * np.log2(q) if q > 0 else 0.0
                c = D[k - 1, i] + pad + lam * ent
                if c < best:
                    best = c
                    bi = i
            D[k, j] = best
            back[k, j] = bi
    return D, back


def _cells(lengths, weights, top, granule):
    lengths = np.asarray(lengths, np.int64)
    if weights is None:
        weights = np.ones(len(lengths))
    weights = np.asarray(weights, float) / np.sum(weights)
    g = max(1, int(granule))
    ntop = int(np.ceil(top / g))
    tops = np.minimum(np.arange(1, ntop + 1) * g, top).astype(np.int64)
    tops[-1] = top
    cell = np.clip(np.ceil(lengths / g).astype(np.int64) - 1, 0, ntop - 1)
    mass = np.bincount(cell, weights=weights, minlength=ntop)
    lsum = np.bincount(cell, weights=weights * lengths, minlength=ntop)
    return tops, mass, lsum


def design(lengths, top, kmax, lam=0.0, granule=16, weights=None):
    """Return bucket tops (ascending, last == top) minimising overhead + lam*H with <= kmax buckets."""
    assert np.max(lengths) <= top
    tops, mass, lsum = _cells(lengths, weights, top, granule)
    D, back = _dp(tops.astype(float), mass, lsum, kmax, float(lam))
    M = len(tops)
    k = int(np.argmin(D[1:, M])) + 1
    # reconstruct
    b = []
    j = M
    while k > 0:
        b.append(int(tops[j - 1]))
        j = int(back[k, j])
        k -= 1
    return np.array(sorted(b), np.int64)


def quantise(lengths, buckets):
    """bucket index and padded size for each length"""
    idx = np.searchsorted(buckets, lengths, side="left")
    return idx, buckets[idx]


def stats(lengths, buckets, weights=None):
    """expected overhead (bytes and relative), bucket entropy H(B) in bits, K"""
    lengths = np.asarray(lengths, np.int64)
    w = np.ones(len(lengths)) if weights is None else np.asarray(weights, float)
    w = w / w.sum()
    idx, padded = quantise(lengths, buckets)
    p = np.bincount(idx, weights=w, minlength=len(buckets))
    p = p[p > 0]
    H = float(-(p * np.log2(p)).sum())
    return dict(pad_bytes=float((w * (padded - lengths)).sum()),
                mean_len=float((w * lengths).sum()),
                mean_padded=float((w * padded).sum()),
                H=H, K=len(buckets), log2K=float(np.log2(len(buckets))))


def _reconstruct(tops, back, k, M):
    b = []
    j = M
    while k > 0:
        b.append(int(tops[j - 1]))
        j = int(back[k, j])
        k -= 1
    return np.array(sorted(b), np.int64)


def design_entropy(lengths, top, kmax, h_target, granule=16, weights=None, lams=None):
    """Least-overhead bucket set with profile entropy H(B) <= h_target and K <= kmax.
    One DP per lambda yields the optimum for every bucket count k <= kmax at once;
    sweeping lambda over a geometric grid and k = 1..kmax traces the attainable
    (entropy, overhead) frontier, from which the best feasible point is taken."""
    tops, mass, lsum = _cells(lengths, weights, top, granule)
    M = len(tops)
    if lams is None:
        lams = np.concatenate([[0.0], np.geomspace(1e-2, 1e6, 48)]) * top / 1000.0
    best, bestc = np.array([top], np.int64), float(top)
    for lam in lams:
        D, back = _dp(tops.astype(float), mass, lsum, kmax, float(lam))
        for k in range(1, kmax + 1):
            if D[k, M] >= 1e299:
                continue
            b = _reconstruct(tops, back, k, M)
            st = stats(lengths, b, weights)
            if st["H"] <= h_target + 1e-12 and st["mean_padded"] < bestc:
                best, bestc = b, st["mean_padded"]
    return best


def pow2_buckets(top, lmin=16):
    """baseline: power-of-two buckets up to top"""
    b = []
    v = lmin
    while v < top:
        b.append(v)
        v *= 2
    b.append(top)
    return np.array(b, np.int64)


def uniform_buckets(top, k):
    """baseline: K equal-width buckets"""
    return np.unique(np.ceil(np.arange(1, k + 1) * top / k).astype(np.int64))
