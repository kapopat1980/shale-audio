import sys, glob, os, json, numpy as np, soundfile as sf
sys.path.insert(0, '.')
from shale import core
FR = json.load(open('results/file_ratios.json'))['K4']
K = os.urandom(16); out = {}
for name, pat in (('crema', 'data/CREMA-D/AudioWAV/*.wav'), ('noisetest', 'data/MS-SNSD/noise_test/*.wav'),
                  ('bach16', 'data/bach16/*.wav'), ('bach24', 'data/bach24/*.wav')):
    ok = n = 0
    for f in sorted(glob.glob(pat)):
        info = sf.info(f); bits = 24 if info.subtype == 'PCM_24' else 16
        x, sr = sf.read(f, dtype='int32'); x = (x >> 16) if bits == 16 else (x >> 8)
        if x.ndim == 1: x = x[:, None]
        x = x.astype(np.int64)
        blob, _ = core.encrypt_file(K, x, sr, bits, 4096, FR)
        ok += np.array_equal(core.decrypt_file(K, blob, FR), x); n += 1
    out[name] = [ok, n]; print(name, ok, n, flush=True)
json.dump(out, open('results/roundtrip_all.json', 'w'))
