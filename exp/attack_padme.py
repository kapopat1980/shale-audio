"""Padmé (Nikitin et al., PETS 2019) as a padding baseline on SHALE records: stream (20/100 ms) and file mode."""
import sys, json, numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'exp')
import attack as A, rocket
from attack_real import padme
from shale import core, eclq
R = 'results'
out = {}
rng = np.random.default_rng(0)
F = 4096
T = np.array([8 + d[F].sum() for d in A.crema]); tops = np.array([core.file_top(n, 1, 16, F) for n in A.N])
P = np.minimum(padme(T), tops)
u, idx = np.unique(P, return_inverse=True); p = np.bincount(idx) / len(idx)
X = np.column_stack([A.N, P])
r = dict(size=float(P.mean() / (2 * A.N).mean()), H=float(-(p * np.log2(p)).sum()), tasks={t: A.cv_eval(A.hgb, X, y, g) for t, (y, g) in A.TASKS.items()})
out['file'] = r; print('file padme', r['size'], r['H'], {t: round(v[2], 3) for t, v in r['tasks'].items()}, flush=True)
json.dump(out, open(f'{R}/attack_padme.json', 'w'), indent=1, default=float)
for Fs in (1600, 320):
    top = core.frame_top(Fs, 1, 16)
    seqs = [np.minimum(padme(d[Fs]), top) for d in A.crema]
    allp = np.concatenate(seqs); u, idx = np.unique(allp, return_inverse=True); p = np.bincount(idx) / len(idx)
    Xs = A.summary_feats(seqs, A.N); Xq, Lq = A.seq_matrix(seqs)
    kern = rocket.make_kernels(500, rng, max(8, Xq.shape[1]))
    Xr = np.hstack([rocket.transform(Xq, Lq, *kern), A.N[:, None] / 16000.0, Lq[:, None]])
    r = dict(size=float((allp.mean() + 16) / (top - 4)), H=float(-(p * np.log2(p)).sum()), tasks={})
    for t, (y, g) in A.TASKS.items():
        r['tasks'][t] = dict(hgb=A.cv_eval(A.hgb, Xs, y, g), rocket=A.cv_eval(A.ridge, Xr, y, g))
    out[f'stream_{Fs}'] = r
    print(Fs, 'padme', r['size'], r['H'], {t: round(max(v['hgb'][2], v['rocket'][2]), 3) for t, v in r['tasks'].items()}, flush=True)
    json.dump(out, open(f'{R}/attack_padme.json', 'w'), indent=1, default=float)
