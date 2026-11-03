"""Numba vs C coder speed (identical bitstream) and end-to-end SHALE file mode with the C coder."""
import sys, json, time, numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'exp')
from shale import codec, ccodec, core
from bench import load
FR = json.load(open('results/file_ratios.json'))['K4']; K = bytes(range(16)); out = {}
for name, pat in (('speech', 'data/CREMA-D/AudioWAV/*.wav'), ('music16', 'data/bach16/*.wav'), ('music24', 'data/bach24/*.wav')):
    x, sr, bits = load(pat, 60); raw = x.size * bits // 8; r = {}
    for lab, enc, dec in (('numba', codec.encode_frames, codec.decode_frames), ('C', ccodec.encode_frames, ccodec.decode_frames)):
        enc(x[:5000], bits, 4096); te = []; td = []
        for _ in range(5):
            t = time.perf_counter(); fr = enc(x, bits, 4096); te.append(time.perf_counter() - t)
            t = time.perf_counter(); y = dec(fr, len(x), x.shape[1], bits, 4096); td.append(time.perf_counter() - t)
        assert np.array_equal(y, x)
        r[lab] = dict(enc=raw / np.median(te) / 1e6, dec=raw / np.median(td) / 1e6)
    core.USE_C = True; te = []; td = []
    for _ in range(5):
        t = time.perf_counter(); blob, j = core.encrypt_file(K, x, sr, bits, 4096, FR); te.append(time.perf_counter() - t)
        t = time.perf_counter(); y = core.decrypt_file(K, blob, FR); td.append(time.perf_counter() - t)
    core.USE_C = False; assert np.array_equal(y, x)
    r['SHALE file (C coder, ascon)'] = dict(enc=raw / np.median(te) / 1e6, dec=raw / np.median(td) / 1e6, size=len(blob) / raw)
    out[name] = r; print(name, r)
json.dump(out, open('results/ccodec_bench.json', 'w'), indent=1)
