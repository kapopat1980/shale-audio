"""Compute SHALE codec record lengths (4-byte field + payload) per frame for corpora."""
import sys, os, glob, pickle, time, numpy as np, soundfile as sf
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, '.')
from shale import core

FRAMES = [320, 1600, 4096]   # 20 ms, 100 ms at 16 kHz; 4096 = file-mode frame

def one(f):
    info = sf.info(f); bits = 24 if info.subtype == "PCM_24" else 16
    x, sr = sf.read(f, dtype="int32")
    if bits == 16: x = x >> 16
    else: x = x >> 8
    x = x.astype(np.int64)
    if x.ndim == 1: x = x[:, None]
    out = dict(file=os.path.basename(f), N=len(x), sr=sr, C=x.shape[1], bits=bits)
    for F in FRAMES:
        p = core.frame_payloads(x, bits, F)
        out[F] = np.array([4 + len(b) for _, b in p], np.int32)
    return out

if __name__ == '__main__':
    name, pattern, limit = sys.argv[1], sys.argv[2], int(sys.argv[3])
    fs = sorted(glob.glob(pattern))
    if limit > 0:
        rng = np.random.default_rng(7); fs = sorted(rng.choice(fs, min(limit, len(fs)), replace=False))
    t = time.time()
    one(fs[0])
    with ProcessPoolExecutor(2) as ex:
        res = list(ex.map(one, fs, chunksize=16))
    pickle.dump(res, open(f'results/len_{name}.pkl', 'wb'))
    print(name, len(res), 'files', round(time.time() - t, 1), 's')
