"""
Length side-channel attack on CREMA-D.

The attacker observes only what an eavesdropper on the ciphertext sees:
  pcm      : number of samples N (any PCM-domain cipher, incl. PCM+AEAD and chaos ciphers)
  stream-* : the sequence of per-frame ciphertext record sizes (+N)
  file-*   : the total ciphertext size (+N)
and tries to infer the spoken sentence (12), the emotion (6), the speaker sex (2) and
the speaker identity (91).

Attackers: (A1) duration-only gradient boosting, (A2) gradient boosting on summary
features of the size sequence, (A3) ROCKET random-convolution features + ridge.
Splits: 5-fold GroupKFold by actor (sentence, emotion, sex) or by sentence (speaker).
"""
import sys, os, pickle, json, time, numpy as np
sys.path.insert(0, '.')
sys.path.insert(0, 'exp')
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.stats import binomtest
from shale import eclq, core
import rocket

R = 'results'
crema = pickle.load(open(f'{R}/len_crema.pkl', 'rb'))
vctk = pickle.load(open(f'{R}/len_vctk.pkl', 'rb'))
demo = {}
for line in open('data/CREMA-D/VideoDemographics.csv').read().splitlines()[1:]:
    a = line.split(',')
    demo[a[0]] = a[2]
files = [d['file'] for d in crema]
actor = np.array([f.split('_')[0] for f in files])
sent = np.array([f.split('_')[1] for f in files])
emo = np.array([f.split('_')[2] for f in files])
sex = np.array([demo[a] for a in actor])
N = np.array([d['N'] for d in crema])
TASKS = dict(sentence=(sent, actor), emotion=(emo, actor), sex=(sex, actor), speaker=(actor, sent))


def schemes_for(F):
    top = core.frame_top(F, 1, 16)
    prof = np.concatenate([d[F] for d in vctk])
    g = 4 if F <= 400 else 16
    S = {'none': None, 'pow2': eclq.pow2_buckets(top), 'uniform4': eclq.uniform_buckets(top, 4)}
    for K in (2, 4, 8, 16):
        S[f'eclq_K{K}'] = eclq.design(prof, top, K, 0.0, g)
    return S, prof, top


def summary_feats(seqs, Nv):
    out = []
    for s, n in zip(seqs, Nv):
        s = s.astype(float)
        q = np.quantile(s, [0, .1, .25, .5, .75, .9, 1])
        prof = np.interp(np.linspace(0, len(s) - 1, 24), np.arange(len(s)), s)
        d = np.diff(s) if len(s) > 1 else np.zeros(1)
        out.append(np.concatenate([[n, len(s), s.mean(), s.std(), s.sum(), np.abs(d).mean(), (d != 0).mean()], q, prof]))
    return np.array(out)


def seq_matrix(seqs):
    T = max(len(s) for s in seqs)
    X = np.zeros((len(seqs), T))
    L = np.zeros(len(seqs), np.int64)
    for i, s in enumerate(seqs):
        s = s.astype(float)
        sd = s.std() if s.std() > 0 else 1.0
        X[i, :len(s)] = (s - s.mean()) / sd
        L[i] = len(s)
    return X, L


def cv_eval(make_model, Xf, y, groups, nfold=5):
    gkf = GroupKFold(nfold)
    correct = 0
    total = 0
    accs = []
    for tr, te in gkf.split(Xf, y, groups):
        m = make_model()
        m.fit(Xf[tr], y[tr])
        p = m.predict(Xf[te])
        c = (p == y[te]).sum()
        correct += c
        total += len(te)
        accs.append(c / len(te))
    return correct, total, float(np.mean(accs)), float(np.std(accs))


def hgb():
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, early_stopping=True,
                                          validation_fraction=0.1, n_iter_no_change=15, random_state=0)


def ridge():
    return make_pipeline(StandardScaler(), RidgeClassifierCV(alphas=np.logspace(-1, 3, 5)))


def run(F, which='all'):
    S, prof, top = schemes_for(F)
    rng = np.random.default_rng(0)
    res = {}
    kern = None
    for name, b in S.items():
        if b is None:
            seqs = [d[F] for d in crema]
        else:
            seqs = [b[np.searchsorted(b, d[F])] for d in crema]
        st = eclq.stats(np.concatenate([d[F] for d in crema]), b if b is not None else np.unique(np.concatenate([d[F] for d in crema])))
        stp = eclq.stats(prof, b) if b is not None else None
        Xs = summary_feats(seqs, N)
        Xq, Lq = seq_matrix(seqs)
        if kern is None:
            kern = rocket.make_kernels(500, rng, max(8, Xq.shape[1]))
        Xr = np.hstack([rocket.transform(Xq, Lq, *kern), N[:, None] / 16000.0, Lq[:, None]])
        r = dict(K=len(b) if b is not None else None, buckets=None if b is None else b.tolist(),
                 H_crema=st['H'], H_profile=None if stp is None else stp['H'],
                 mean_padded=st['mean_padded'], mean_len=st['mean_len'], top=top, tasks={})
        for t, (y, grp) in TASKS.items():
            a2 = cv_eval(hgb, Xs, y, grp)
            a3 = cv_eval(ridge, Xr, y, grp)
            r['tasks'][t] = dict(hgb=a2, rocket=a3)
            print(F, name, t, 'hgb %.3f' % a2[2], 'rocket %.3f' % a3[2], flush=True)
        res[name] = r
    return res


def run_duration():
    Xd = N[:, None].astype(float)
    out = {}
    for t, (y, grp) in TASKS.items():
        out[t] = cv_eval(hgb, Xd, y, grp)
        print('duration', t, '%.3f' % out[t][2], flush=True)
    return out


def run_file(F=4096):
    """file mode: attacker sees total plaintext size (unpadded or bucketed) + N"""
    T = np.array([8 + d[F].sum() for d in crema])
    tops = np.array([core.file_top(n, 1, 16, F) for n in N])
    rv = np.array([8 + d[F].sum() for d in vctk]) / np.array([core.file_top(d['N'], 1, 16, F) for d in vctk])
    out = {}
    U = 10000
    lv = np.ceil(rv * U).astype(np.int64)
    designs = {'none': None}
    for K in (2, 4, 8):
        designs[f'eclq_K{K}'] = eclq.design(lv, U, K, 0.0, 10) / U
    for name, rat in designs.items():
        if rat is None:
            obs = T.astype(float)
            Hc = None
            size = T.mean()
        else:
            idx = np.array([core.file_padded_size(t, n, 1, 16, F, rat)[1] for t, n in zip(T, N)])
            sizes = np.array([core.file_padded_size(t, n, 1, 16, F, rat)[0] for t, n in zip(T, N)])
            obs = idx.astype(float)
            p = np.bincount(idx) / len(idx)
            p = p[p > 0]
            Hc = float(-(p * np.log2(p)).sum())
            size = sizes.mean()
        X = np.column_stack([N, obs])
        rr = dict(ratios=None if rat is None else rat.tolist(), H_crema=Hc, mean_size=float(size),
                  mean_raw=float((2 * N).mean()), tasks={})
        for t, (y, grp) in TASKS.items():
            rr['tasks'][t] = cv_eval(hgb, X, y, grp)
            print('file', name, t, '%.3f' % rr['tasks'][t][2], flush=True)
        out[name] = rr
    return out


if __name__ == '__main__':
    t0 = time.time()
    out = dict(duration=run_duration(), file=run_file())
    dump = lambda: json.dump(out, open(f'{R}/attack.json', 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    dump()
    for F in (1600, 320):
        out[f'stream_{F}'] = run(F)
        dump()
    out['classes'] = {t: int(len(np.unique(y))) for t, (y, g) in TASKS.items()}
    out['majority'] = {t: float(np.max(np.unique(y, return_counts=True)[1]) / len(y)) for t, (y, g) in TASKS.items()}
    json.dump(out, open(f'{R}/attack.json', 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    print('done', time.time() - t0)
