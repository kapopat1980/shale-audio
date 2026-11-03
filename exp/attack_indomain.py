"""In-domain profile: buckets designed on 45 CREMA-D actors, attack evaluated on the other 46 (actor-grouped CV)."""
import sys, json, numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'exp')
import attack as A
from shale import eclq, core
rng = np.random.default_rng(5)
actors = np.unique(A.actor); rng.shuffle(actors)
design_set = set(actors[:45])
D = np.array([a in design_set for a in A.actor]); E = ~D
out = dict(design_actors=sorted(design_set))
crema = A.crema; N = A.N
# ---------- file mode
F = 4096
T = np.array([8 + d[F].sum() for d in crema]); tops = np.array([core.file_top(n, 1, 16, F) for n in N])
U = 10000; lv = np.ceil(T[D] / tops[D] * U).astype(np.int64)
designs = {'none': None, 'K1': np.array([1.0])}
for K in (2, 4): designs[f'K{K}'] = eclq.design(lv, U, K, 0.0, 10) / U
for h in (0.05, 0.1, 0.2, 0.35): designs[f'H{h}'] = eclq.design_entropy(lv, U, 4, h, 10) / U
fm = {}
for name, rat in designs.items():
    if rat is None:
        obs = T[E].astype(float); size = T[E].mean(); Hc = None
    else:
        res = [core.file_padded_size(t, n, 1, 16, F, rat) for t, n in zip(T[E], N[E])]
        obs = np.array([r[1] for r in res], float); size = np.mean([r[0] for r in res])
        p = np.bincount(obs.astype(int)) / len(obs); p = p[p > 0]; Hc = float(-(p * np.log2(p)).sum())
    X = np.column_stack([N[E], obs]); r = dict(ratios=None if rat is None else rat.tolist(), H_eval=Hc, size=float(size / (2 * N[E]).mean()), tasks={})
    for t, (y, g) in A.TASKS.items():
        r['tasks'][t] = A.cv_eval(A.hgb, X, y[E], g[E])
    fm[name] = r
    print('file', name, None if rat is None else np.round(rat, 3), 'H', Hc, 'size %.3f' % r['size'], {t: round(v[2], 3) for t, v in r['tasks'].items()}, flush=True)
Xd = N[E][:, None].astype(float)
fm['duration'] = dict(tasks={t: A.cv_eval(A.hgb, Xd, y[E], g[E]) for t, (y, g) in A.TASKS.items()})
print('duration', {t: round(v[2], 3) for t, v in fm['duration']['tasks'].items()}, flush=True)
out['file'] = fm
json.dump(out, open('results/attack_indomain.json', 'w'), indent=1, default=float)
