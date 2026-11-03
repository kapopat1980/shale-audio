"""Paired tests for the file-mode claims: per-clip predictions of the duration-only attacker vs the
attacker that also sees the bucket index; McNemar exact test + bootstrap CI of the accuracy gain."""
import sys, json, numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'exp')
import attack as A
from shale import eclq, core
from scipy.stats import binomtest
from sklearn.model_selection import GroupKFold

from sklearn.ensemble import HistGradientBoostingClassifier as HGB
def preds(X, y, g):
    p = np.empty(len(y), object)
    many = len(np.unique(y)) > 20
    for tr, te in GroupKFold(5).split(X, y, g):
        m = HGB(max_iter=40, learning_rate=0.1, random_state=0) if many else A.hgb()
        p[te] = m.fit(X[tr], y[tr]).predict(X[te])
    return p == y

def compare(cd, cb, rng):
    gain = cb.mean() - cd.mean()
    b = int(np.sum(cb & ~cd)); c = int(np.sum(~cb & cd))
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    bs = [np.mean(cb[i]) - np.mean(cd[i]) for i in (rng.integers(0, len(cb), len(cb)) for _ in range(2000))]
    return dict(acc_dur=float(cd.mean()), acc=float(cb.mean()), gain=float(gain), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], mcnemar_p=float(p), b=b, c=c)

F = 4096; N = A.N
T = np.array([8 + d[F].sum() for d in A.crema])
ind = json.load(open('results/attack_indomain.json'))
fx = json.load(open('results/attack_file_extra.json'))
Dm = np.array([a in set(ind['design_actors']) for a in A.actor]); E = ~Dm
rng = np.random.default_rng(0)
out = {}
def run(label, ratios, mask):
    idx = np.array([core.file_padded_size(t, n, 1, 16, F, ratios)[1] if ratios is not None else t for t, n in zip(T[mask], N[mask])], float)
    out[label] = {}
    for t, (y, g) in A.TASKS.items():
        cd = preds(N[mask][:, None].astype(float), y[mask], g[mask])
        cb = preds(np.column_stack([N[mask], idx]), y[mask], g[mask])
        out[label][t] = compare(cd, cb, rng)
        r = out[label][t]
        print(label, t, 'dur %.3f with %.3f gain %+.3f CI [%+.3f, %+.3f] p=%.3g' % (r['acc_dur'], r['acc'], r['gain'], r['ci'][0], r['ci'][1], r['mcnemar_p']), flush=True)
all_ = np.ones(len(N), bool)
run('public none', None, all_)
for k in ('H0.05', 'H0.1'):
    run('public ' + k, np.array(fx[k]['ratios']), all_)
for k in ('none', 'K2', 'H0.05', 'H0.1', 'H0.2'):
    run('in-domain ' + k, None if ind['file'][k]['ratios'] is None else np.array(ind['file'][k]['ratios']), E)
json.dump(out, open('results/significance.json', 'w'), indent=1)
