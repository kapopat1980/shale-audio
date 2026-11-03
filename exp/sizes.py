import sys, pickle, json, glob, subprocess, os, numpy as np
sys.path.insert(0, '.')
from shale import core
R = 'results'
FR = json.load(open(f'{R}/file_ratios.json'))
rng = np.random.default_rng(3)
sets = {'CREMA-D speech': ('crema', 'data/CREMA-D/AudioWAV/'), 'MS-SNSD noise (test)': ('noisetest', 'data/MS-SNSD/noise_test/'),
        'Bach10 stereo 16-bit': ('bach16', 'data/bach16/'), 'Bach10 stereo 24-bit': ('bach24', 'data/bach24/')}
out = {}
for name, (key, d) in sets.items():
    L = pickle.load(open(f'{R}/len_{key}.pkl', 'rb'))
    raw = np.array([x['N'] * x['C'] * x.get('bits', 16) // 8 for x in L], float)
    T = np.array([8 + x[4096].sum() for x in L], float)
    r = dict(files=len(L), hours=float(sum(x['N'] / x['sr'] for x in L) / 3600), codec=float(T.sum() / raw.sum()))
    for K in ('K2', 'K4', 'K8'):
        s = [core.file_padded_size(int(t), x['N'], x['C'], x.get('bits', 16), 4096, FR[K])[0] + 16 * (-(-1 // 1)) for t, x in zip(T, L)]
        r['file_' + K] = float((np.array(s) + 49 + 16 * np.ceil(np.array(s) / 65536)).sum() / raw.sum())
    sub = L if len(L) < 100 else [L[i] for i in rng.choice(len(L), 400, replace=False)]
    fr5 = fr8 = rw = 0
    for x in sub:
        f = d + x['file']
        rw += x['N'] * x['C'] * x.get('bits', 16) // 8
        for lv in (5, 8):
            n = len(subprocess.run(['ffmpeg', '-v', 'error', '-i', f, '-f', 'flac', '-compression_level', str(lv), '-'], capture_output=True).stdout)
            if lv == 5: fr5 += n
            else: fr8 += n
    Ts = sum(8 + x[4096].sum() for x in sub)
    r.update(flac5=fr5 / rw, flac8=fr8 / rw, codec_sub=Ts / rw, flac_files=len(sub))
    out[name] = r
    print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
json.dump(out, open(f'{R}/sizes.json', 'w'), indent=1)
