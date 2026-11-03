"""Length attack on real codecs (FFmpeg FLAC, libopus VBR/CBR) and ECLQ / Padmé applied to them."""
import sys, json, pickle, numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'exp')
import attack as A, rocket
from shale import eclq, core

def padme(L):
    L = np.asarray(L, np.int64); out = L.copy()
    for i, l in enumerate(L):
        if l < 2: continue
        E = int(np.floor(np.log2(l))); S = int(np.floor(np.log2(E))) + 1 if E > 0 else 1
        m = (1 << max(E - S, 0)) - 1; out[i] = (l + m) & ~m
    return out

if __name__ != '__main__':
    pass
R = 'results'
which = sys.argv[1] if len(sys.argv) > 1 and __name__ == "__main__" else ""
if __name__ == '__main__':
    rc = {d['file']: d for d in pickle.load(open(f'{R}/real_crema.pkl', 'rb'))}
    rv = pickle.load(open(f'{R}/real_vctk.pkl', 'rb'))
    out = json.load(open(f'{R}/attack_real.json')) if __import__('os').path.exists(f'{R}/attack_real.json') else {}
    rng = np.random.default_rng(0)

    def evaluate(name, seqs, info):
        if name in out: return
        Xs = A.summary_feats(seqs, A.N); Xq, Lq = A.seq_matrix(seqs)
        kern = rocket.make_kernels(500, rng, max(8, Xq.shape[1]))
        Xr = np.hstack([rocket.transform(Xq, Lq, *kern), A.N[:, None] / 16000.0, Lq[:, None]])
        r = dict(info, tasks={})
        for t, (y, g) in A.TASKS.items():
            r['tasks'][t] = dict(hgb=A.cv_eval(A.hgb, Xs, y, g), rocket=A.cv_eval(A.ridge, Xr, y, g))
        out[name] = r
        json.dump(out, open(f'{R}/attack_real.json', 'w'), indent=1, default=float)
        print(name, info, {t: round(max(v['hgb'][2], v['rocket'][2]), 3) for t, v in r['tasks'].items()}, flush=True)

    for codec_key in which.split(','):
        seqs = [rc[f][codec_key] for f in A.files]
        allL = np.concatenate(seqs)
        if codec_key.startswith('opus'):
            ref = 24000 / 8 * 0.020          # nominal CBR packet at 24 kb/s, 20 ms = 60 bytes
        else:
            F = 320 if '20ms' in codec_key else 1600
            ref = 2 * F
        evaluate(codec_key + '|none', seqs, dict(size=float(allL.mean() / ref)))
        if codec_key in ('opus_vbr24_20ms', 'flac_20ms'):
            top = int(max(allL.max(), np.concatenate([d[codec_key] for d in rv]).max()))
            prof = np.concatenate([d[codec_key] for d in rv])
            evaluate(codec_key + '|padme', [np.minimum(padme(s), top) for s in seqs], dict(size=float(np.concatenate([padme(s) for s in seqs]).mean() / ref)))
            for h in (0.1, 0.3):
                b = eclq.design_entropy(prof, top, 4, h, 1 if top < 400 else 4)
                ps = [b[np.searchsorted(b, s)] for s in seqs]
                st = eclq.stats(allL, b)
                evaluate(f'{codec_key}|eclq_H{h}', ps, dict(size=float(st['mean_padded'] / ref), H=st['H'], buckets=b.tolist(), top=top))
