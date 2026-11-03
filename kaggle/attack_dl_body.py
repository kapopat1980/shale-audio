# ======================================================================
# SHALE paper - deep-learning length side-channel attack (Kaggle, GPU)
# Inputs (Add Input in Kaggle): CREMA-D audio and Google Speech Commands.
# Output: /kaggle/working/shale_dl_results.json  (send this file back)
# ======================================================================
import os, re, glob, json, time, math, random, sys
import numpy as np
import soundfile as sf
import torch, torch.nn as nn, torch.nn.functional as Fnn
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, '/kaggle/working/shale_mod')
import codec, eclq

QUICK = False          # True = small smoke test (few files, 2 epochs)
NFOLD = 5
EPOCHS = 40
OUT = '/kaggle/working/shale_dl_results.json'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device', DEV)
C = CONST
torch.manual_seed(0); np.random.seed(0); random.seed(0)

# ---------------------------------------------------------------- data discovery
allwav = glob.glob('/kaggle/input/**/*.wav', recursive=True)
crema = sorted({os.path.basename(f): f for f in allwav if re.match(r'^\d{4}_[A-Z]{3}_[A-Z]{3}_[A-Z]{2}\.wav$', os.path.basename(f))}.values())
sc = [f for f in allwav if re.match(r'^[0-9a-f]{8}_nohash_\d+\.wav$', os.path.basename(f))
      and '_background_noise_' not in f]
print('CREMA-D files', len(crema), ' Speech Commands files', len(sc))

def record_lengths(f, frames=(320, 1600, 4096)):
    x, sr = sf.read(f, dtype='int16')
    if x.ndim > 1: x = x[:, 0]
    if sr != 16000: return None
    x = x.astype(np.int64)[:, None]
    out = dict(N=len(x))
    for F in frames:
        fr = codec.encode_frames(x, 16, F)
        L = []
        for i, fb in enumerate(fr):
            raw = 2 * min(F, len(x) - i * F)
            L.append(4 + (raw if len(fb) >= raw else len(fb)))     # verbatim fallback
        out[F] = np.array(L, np.int32)
    return out

def lengths_for(files):
    codec.encode_frames(np.zeros((4000, 1), np.int64), 16, 320)   # JIT warm-up
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        return list(ex.map(record_lengths, files, chunksize=32))

# ---------------------------------------------------------------- padding schemes
def padme(L):
    L = np.asarray(L, np.int64); out = L.copy()
    for i, l in enumerate(L):
        if l < 2: continue
        E = int(math.floor(math.log2(l))); S = int(math.floor(math.log2(E))) + 1 if E > 0 else 1
        z = max(E - S, 0); m = (1 << z) - 1
        out[i] = (l + m) & ~m
    return out

def schemes(F, top, indomain_prof):
    B = {k: np.array(v) for k, v in C['buckets'][str(F)].items()}
    S = {'none': lambda L: L,
         'padme': lambda L: np.minimum(padme(L), top),
         'pow2': lambda L, b=eclq.pow2_buckets(top): b[np.searchsorted(b, L)]}
    for k in ('K2', 'K4', 'H0.1', 'H0.3'):
        S['eclq_' + k] = (lambda L, b=B[k]: b[np.searchsorted(b, L)])
    if indomain_prof is not None:
        b = eclq.design_entropy(indomain_prof, top, 4, 0.1, 4 if F <= 400 else 16)
        S['eclq_H0.1_indomain'] = (lambda L, b=b: b[np.searchsorted(b, L)])
    return S

# ---------------------------------------------------------------- models
class CNN(nn.Module):
    def __init__(s, ncls):
        super().__init__()
        s.c = nn.Sequential(nn.Conv1d(2, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
                            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
                            nn.Conv1d(128, 128, 5, padding=2, dilation=1), nn.BatchNorm1d(128), nn.ReLU(),
                            nn.Conv1d(128, 128, 3, padding=2, dilation=2), nn.BatchNorm1d(128), nn.ReLU())
        s.h = nn.Sequential(nn.Linear(256 + 2, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, ncls))
    def forward(s, x, m, d):
        z = s.c(torch.stack([x, m], 1))
        mm = Fnn.max_pool1d(m[:, None], 2) if z.shape[-1] != m.shape[-1] else m[:, None]
        mm = mm[..., :z.shape[-1]]
        avg = (z * mm).sum(-1) / mm.sum(-1).clamp(min=1)
        mx = (z - 1e4 * (1 - mm)).max(-1).values
        return s.h(torch.cat([avg, mx, d], 1))

class TRF(nn.Module):
    def __init__(s, ncls, T):
        super().__init__()
        s.e = nn.Linear(1, 64); s.p = nn.Parameter(torch.randn(1, T, 64) * 0.02)
        s.t = nn.TransformerEncoder(nn.TransformerEncoderLayer(64, 4, 128, 0.1, batch_first=True), 2)
        s.h = nn.Sequential(nn.Linear(64 + 2, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, ncls))
    def forward(s, x, m, d):
        z = s.t(s.e(x[..., None]) + s.p[:, :x.shape[1]], src_key_padding_mask=(m == 0))
        avg = (z * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp(min=1)
        return s.h(torch.cat([avg, d], 1))

def to_tensors(seqs, N, top):
    T = max(len(s) for s in seqs)
    X = np.zeros((len(seqs), T), np.float32); M = np.zeros_like(X)
    for i, s in enumerate(seqs):
        X[i, :len(s)] = np.asarray(s, np.float32) / top; M[i, :len(s)] = 1
    D = np.stack([N / 80000.0, np.array([len(s) for s in seqs]) / T], 1).astype(np.float32)
    return torch.tensor(X), torch.tensor(M), torch.tensor(D)

def fit_eval(kind, X, M, D, y, tr, te, ncls, groups):
    # inner validation split by group for early stopping
    g = groups[tr]; ug = np.unique(g); rng = np.random.default_rng(1); vg = set(rng.choice(ug, max(1, len(ug) // 8), replace=False))
    va = tr[np.isin(g, list(vg))]; trn = tr[~np.isin(g, list(vg))]
    model = (CNN(ncls) if kind == 'cnn' else TRF(ncls, X.shape[1])).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), 2e-3, weight_decay=1e-2)
    E = 2 if QUICK else EPOCHS
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=E * math.ceil(len(trn) / 128))
    yt = torch.tensor(y)
    best, bstate, bad = -1, None, 0
    for ep in range(E):
        model.train(); perm = np.random.permutation(trn)
        for i in range(0, len(perm), 128):
            b = perm[i:i + 128]
            loss = Fnn.cross_entropy(model(X[b].to(DEV), M[b].to(DEV), D[b].to(DEV)), yt[b].to(DEV), label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        acc = predict(model, X, M, D, va).eq(yt[va]).float().mean().item()
        if acc > best: best, bstate, bad = acc, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else: bad += 1
        if bad >= 8: break
    model.load_state_dict(bstate)
    p = predict(model, X, M, D, te)
    return int(p.eq(yt[te]).sum().item())

@torch.no_grad()
def predict(model, X, M, D, idx):
    model.eval(); out = []
    for i in range(0, len(idx), 512):
        b = idx[i:i + 512]
        out.append(model(X[b].to(DEV), M[b].to(DEV), D[b].to(DEV)).argmax(1).cpu())
    return torch.cat(out)

def cv(seqs, N, top, y, groups, models=('cnn', 'trf')):
    labs, yi = np.unique(y, return_inverse=True)
    X, M, D = to_tensors(seqs, N, top)
    res = {}
    for kind in models:
        correct = 0; total = 0; folds = []
        for tr, te in GroupKFold(min(NFOLD, len(np.unique(groups)))).split(np.zeros(len(y)), yi, groups):
            c = fit_eval(kind, X, M, D, yi, tr, te, len(labs), groups)
            correct += c; total += len(te); folds.append(c / len(te))
        res[kind] = [correct, total, float(np.mean(folds)), float(np.std(folds))]
    return res

def cv_duration(N, y, groups):
    labs, yi = np.unique(y, return_inverse=True); correct = 0; folds = []
    for tr, te in GroupKFold(min(NFOLD, len(np.unique(groups)))).split(N[:, None], yi, groups):
        m = HistGradientBoostingClassifier(max_iter=200, early_stopping=True, random_state=0).fit(N[tr, None], yi[tr])
        c = int((m.predict(N[te, None]) == yi[te]).sum()); correct += c; folds.append(c / len(te))
    return [correct, len(y), float(np.mean(folds)), float(np.std(folds))]

R = json.load(open(OUT)) if os.path.exists(OUT) else {}
def save(): json.dump(R, open(OUT, 'w'), indent=1)

def run_dataset(name, files, tasks, design_mask):
    if QUICK: files = list(np.random.default_rng(0).choice(files, min(600, len(files)), replace=False))   # spread over all speakers
    t0 = time.time(); L = lengths_for(files); keep = [i for i, l in enumerate(L) if l is not None]
    L = [L[i] for i in keep]; files = [files[i] for i in keep]
    print(name, 'lengths', len(L), round(time.time() - t0), 's', flush=True)
    N = np.array([l['N'] for l in L], np.float64)
    R.setdefault(name, {})
    for tname, (y, g, mask) in tasks(files).items():
        key = f'duration|{tname}'
        if key not in R[name]:
            R[name][key] = cv_duration(N[mask], y[mask], g[mask]); save(); print(name, key, R[name][key], flush=True)
    dm = design_mask(files)
    for F in (1600, 320):
        top = 2 * F + 4
        prof = np.concatenate([L[i][F] for i in np.where(dm)[0]]) if dm is not None else None
        for sname, fn in schemes(F, top, prof).items():
            seqs_all = [fn(l[F]) for l in L]
            allL = np.concatenate([l[F] for l in L]); padded = np.concatenate(seqs_all)
            idx = np.searchsorted(np.unique(padded), padded); p = np.bincount(idx) / len(idx)
            info = dict(size=float((padded.sum() + 16 * len(padded)) / (2 * F * len(padded))), H=float(-(p * np.log2(p)).sum()))
            for tname, (y, g, mask) in tasks(files).items():
                if sname.endswith('indomain'): mask = mask & ~dm
                key = f'{F}|{sname}|{tname}'
                if key in R[name]: continue
                seqs = [seqs_all[i] for i in np.where(mask)[0]]
                r = cv(seqs, N[mask], top, y[mask], g[mask]); r.update(info)
                if sname.endswith('indomain'):
                    r['duration_same_subset'] = cv_duration(N[mask], y[mask], g[mask])
                R[name][key] = r; save(); print(name, key, {k: (round(v[2], 3) if isinstance(v, list) else v) for k, v in r.items()}, flush=True)

# ---------------------------------------------------------------- CREMA-D
if crema:
    ind = set(C['design_actors'])
    def crema_tasks(files):
        b = [os.path.basename(f) for f in files]
        actor = np.array([s.split('_')[0] for s in b]); sent = np.array([s.split('_')[1] for s in b]); emo = np.array([s.split('_')[2] for s in b])
        allm = np.ones(len(b), bool)
        return {'sentence': (sent, actor, allm), 'emotion': (emo, actor, allm), 'speaker': (actor, sent, allm)}
    run_dataset('CREMA-D', crema, crema_tasks, lambda files: np.array([os.path.basename(f).split('_')[0] in ind for f in files]))

# ---------------------------------------------------------------- Speech Commands (35 keywords)
if sc:
    rng = np.random.default_rng(0)
    bylab = {}
    for f in sc: bylab.setdefault(os.path.basename(os.path.dirname(f)), []).append(f)
    sel = []
    for lab, fs in sorted(bylab.items()):
        fs = sorted(set(fs)); rng.shuffle(fs); sel += fs[:(60 if QUICK else 300)]
    def sc_tasks(files):
        lab = np.array([os.path.basename(os.path.dirname(f)) for f in files]); spk = np.array([os.path.basename(f).split('_')[0] for f in files])
        return {'keyword': (lab, spk, np.ones(len(files), bool))}
    def sc_design(files):
        spk = np.array([os.path.basename(f).split('_')[0] for f in files])
        return np.array([int(s[0], 16) < 8 for s in spk])     # half of the speakers design the in-domain profile
    run_dataset('SpeechCommands', sel, sc_tasks, sc_design)
print('ALL DONE ->', OUT)
