"""File-mode attack with entropy-constrained ECLQ designs (speech profile = VCTK)."""
import sys, json, numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'exp')
import attack as A
from shale import eclq, core
F = 4096
T = np.array([8 + d[F].sum() for d in A.crema])
N = A.N
rv = np.array([8 + d[F].sum() for d in A.vctk]) / np.array([core.file_top(d['N'], 1, 16, F) for d in A.vctk])
U = 10000
lv = np.ceil(rv * U).astype(np.int64)
designs = {'K1 (constant)': np.array([1.0])}
for h in (0.05, 0.1, 0.2, 0.35, 0.5):
    designs[f'H{h}'] = eclq.design_entropy(lv, U, 8, h, 10) / U
out = {}
for name, rat in designs.items():
    res = [core.file_padded_size(t, n, 1, 16, F, rat) for t, n in zip(T, N)]
    idx = np.array([r[1] for r in res]); sizes = np.array([r[0] for r in res])
    p = np.bincount(idx) / len(idx); p = p[p > 0]
    Hc = float(-(p * np.log2(p)).sum())
    pv = np.bincount(np.searchsorted(rat, rv, side='left'), minlength=len(rat)) / len(rv); pv = pv[pv > 0]
    X = np.column_stack([N, idx])
    rr = dict(ratios=rat.tolist(), H_crema=Hc, H_profile=float(-(pv * np.log2(pv)).sum()), mean_size=float(sizes.mean()),
              mean_raw=float((2 * N).mean()), tasks={})
    for t, (y, grp) in A.TASKS.items():
        rr['tasks'][t] = A.cv_eval(A.hgb, X, y, grp)
    out[name] = rr
    print(name, np.round(rat, 3), 'H_crema %.3f H_prof %.3f size %.3f' % (Hc, rr['H_profile'], sizes.mean() / (2 * N).mean()),
          {t: round(v[2], 3) for t, v in rr['tasks'].items()}, flush=True)
json.dump(out, open('results/attack_file_extra.json', 'w'), indent=1, default=float)
