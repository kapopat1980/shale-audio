"""SHALE-SIV: bit-exactness, determinism, tamper rejection and sensitivity with a REUSED salt. Writes results/siv.json."""
import sys, os, glob, json, numpy as np, soundfile as sf
sys.path.insert(0, '.')
from shale import core
FR = json.load(open('results/file_ratios.json'))['K4']
rng = np.random.default_rng(11); K = os.urandom(16)
fs = rng.choice(sorted(glob.glob('data/CREMA-D/AudioWAV/*.wav')), 10, replace=False)
npcr = []; uaci = []; nscr = []; u16 = []; ok = same = rej = 0
for f in fs:
    x, sr = sf.read(f, dtype='int16'); x = x.astype(np.int64)[:, None]
    c1, _ = core.encrypt_file_siv(K, x, sr, 16, 4096, FR)
    ok += np.array_equal(core.decrypt_file_siv(K, c1, FR), x)
    same += core.encrypt_file_siv(K, x, sr, 16, 4096, FR)[0] == c1
    for _ in range(4):
        x2 = x.copy(); x2[rng.integers(len(x)), 0] ^= 1
        c2, _ = core.encrypt_file_siv(K, x2, sr, 16, 4096, FR)
        a = np.frombuffer(c1[core.HDR_LEN:], np.uint8); b = np.frombuffer(c2[core.HDR_LEN:], np.uint8); m = min(len(a), len(b)) // 2 * 2
        npcr.append(np.mean(a[:m] != b[:m]) * 100); uaci.append(np.mean(np.abs(a[:m].astype(float) - b[:m])) / 255 * 100)
        aw = np.frombuffer(a[:m].tobytes(), '<u2').astype(float); bw = np.frombuffer(b[:m].tobytes(), '<u2').astype(float)
        nscr.append(np.mean(aw != bw) * 100); u16.append(np.mean(np.abs(aw - bw)) / 65535 * 100)
    bb = bytearray(c1); bb[100] ^= 1
    try: core.decrypt_file_siv(K, bytes(bb), FR)
    except core.AuthError: rej += 1
r = dict(npcr8=float(np.mean(npcr)), uaci8=float(np.mean(uaci)), nscr16=float(np.mean(nscr)), uaci16=float(np.mean(u16)),
         min_nscr16=float(min(nscr)), n=len(npcr), exact=ok, deterministic=same, tamper_rej=rej)
json.dump(r, open('results/siv.json', 'w'), indent=1); print(r)
