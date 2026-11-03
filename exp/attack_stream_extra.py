"""Stream-mode attack with entropy-constrained ECLQ (public VCTK profile and in-domain CREMA profile)."""
import sys, json, numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'exp')
import attack as A, rocket
from shale import eclq, core
F = int(sys.argv[1]); prof_kind = sys.argv[2]; targets = [float(h) for h in sys.argv[3].split(',')]
top = core.frame_top(F, 1, 16); g = 4 if F <= 400 else 16
ind = json.load(open('results/attack_indomain.json'))['design_actors']
Dm = np.array([a in set(ind) for a in A.actor])
if prof_kind == 'vctk':
    prof = np.concatenate([d[F] for d in A.vctk]); E = np.ones(len(A.crema), bool)
else:
    prof = np.concatenate([d[F] for i, d in enumerate(A.crema) if Dm[i]]); E = ~Dm
out = {}
rng = np.random.default_rng(0)
for h in targets:
    b = eclq.design_entropy(prof, top, 4, h, g)
    seqs = [b[np.searchsorted(b, d[F])] for i, d in enumerate(A.crema) if E[i]]
    Nv = A.N[E]
    allL = np.concatenate([d[F] for i, d in enumerate(A.crema) if E[i]])
    st = eclq.stats(allL, b); stp = eclq.stats(prof, b)
    Xs = A.summary_feats(seqs, Nv); Xq, Lq = A.seq_matrix(seqs)
    kern = rocket.make_kernels(500, rng, max(8, Xq.shape[1]))
    Xr = np.hstack([rocket.transform(Xq, Lq, *kern), Nv[:, None] / 16000.0, Lq[:, None]])
    r = dict(buckets=b.tolist(), H_eval=st['H'], H_profile=stp['H'], size=(st['mean_padded'] + 16) / (top - 4), tasks={})
    for t, (y, grp) in A.TASKS.items():
        r['tasks'][t] = dict(hgb=A.cv_eval(A.hgb, Xs, y[E], grp[E]), rocket=A.cv_eval(A.ridge, Xr, y[E], grp[E]))
    out[f'H{h}'] = r
    print(F, prof_kind, h, b.tolist(), 'H_eval %.3f H_prof %.3f size %.3f' % (st['H'], stp['H'], r['size']),
          {t: round(max(v['hgb'][2], v['rocket'][2]), 3) for t, v in r['tasks'].items()}, flush=True)
    json.dump(out, open(f'results/attack_stream_{F}_{prof_kind}.json', 'w'), indent=1, default=float)
