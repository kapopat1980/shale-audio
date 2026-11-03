"""Minimal ROCKET (random convolutional kernel) features for variable-length 1-D sequences."""
import numpy as np
from numba import njit, prange


def make_kernels(n_kernels, rng, max_len):
    lens = rng.choice(np.array([7, 9, 11]), n_kernels)
    W = np.zeros((n_kernels, 11))
    B = np.zeros(n_kernels)
    D = np.zeros(n_kernels, np.int64)
    P = np.zeros(n_kernels, np.int64)
    for k in range(n_kernels):
        L = lens[k]
        w = rng.normal(0, 1, L)
        W[k, :L] = w - w.mean()
        B[k] = rng.uniform(-1, 1)
        A = np.log2((max_len - 1) / (L - 1)) if max_len > L else 0
        D[k] = int(2 ** rng.uniform(0, max(A, 0)))
        P[k] = ((L - 1) * D[k]) // 2 if rng.integers(2) else 0
    return lens.astype(np.int64), W, B, D, P


@njit(parallel=True, cache=True)
def transform(X, lens_x, klen, W, B, D, P):
    n = X.shape[0]
    K = W.shape[0]
    out = np.zeros((n, 2 * K))
    for i in prange(n):
        T = lens_x[i]
        x = X[i, :T]
        for k in range(K):
            L = klen[k]; d = D[k]; p = P[k]
            outlen = T + 2 * p - (L - 1) * d
            mx = -1e300
            ppv = 0
            cnt = 0
            for t in range(outlen):
                s = B[k]
                for j in range(L):
                    idx = t - p + j * d
                    if 0 <= idx < T:
                        s += W[k, j] * x[idx]
                if s > mx:
                    mx = s
                if s > 0:
                    ppv += 1
                cnt += 1
            out[i, 2 * k] = ppv / cnt if cnt > 0 else 0.0
            out[i, 2 * k + 1] = mx if cnt > 0 else 0.0
    return out
