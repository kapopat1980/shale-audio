"""Throughput and size: SHALE vs PCM-domain AEADs on the same machine (one core)."""
import sys, os, glob, time, json, pickle, numpy as np, soundfile as sf
sys.path.insert(0, '.')
os.environ.setdefault('NUMBA_NUM_THREADS', '1')
from shale import core, eclq, ascon, codec
from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305

R = 'results'
K = bytes(range(16))


def load(pattern, seconds, bits_hint=None):
    xs, sr = [], None
    for f in sorted(glob.glob(pattern)):
        info = sf.info(f)
        bits = 24 if info.subtype == 'PCM_24' else 16
        x, sr = sf.read(f, dtype='int32')
        x = (x >> 16) if bits == 16 else (x >> 8)
        if x.ndim == 1:
            x = x[:, None]
        xs.append(x)
        if sum(len(a) for a in xs) >= seconds * sr:
            break
    x = np.concatenate(xs)[: int(seconds * sr)].astype(np.int64)
    return x, sr, bits


def med(fn, reps=5):
    ts = []
    out = None
    for _ in range(reps):
        t = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts)), out


def bench(name, x, sr, bits, F_file, F_stream, stream_buckets, file_ratios):
    N, C = x.shape
    pcm = core.pcm_bytes(x, bits)
    raw = len(pcm)
    res = {'raw_bytes': raw, 'seconds': N / sr, 'sr': sr, 'C': C, 'bits': bits}
    n12 = os.urandom(12)
    # --- PCM-domain AEADs
    for nm, mk in (('PCM + AES-128-GCM', lambda: AESGCM(os.urandom(16))),
                   ('PCM + AES-256-GCM', lambda: AESGCM(os.urandom(32))),
                   ('PCM + ChaCha20-Poly1305', lambda: ChaCha20Poly1305(os.urandom(32)))):
        a = mk()
        te, ct = med(lambda: a.encrypt(n12, pcm, b''))
        td, _ = med(lambda: a.decrypt(n12, ct, b''))
        res[nm] = dict(enc=raw / te / 1e6, dec=raw / td / 1e6, size=len(ct) / raw)
    nn = os.urandom(16)
    te, ct = med(lambda: ascon.encrypt(K, nn, b'', pcm))
    td, _ = med(lambda: ascon.decrypt(K, nn, b'', ct))
    res['PCM + Ascon-AEAD128'] = dict(enc=raw / te / 1e6, dec=raw / td / 1e6, size=len(ct) / raw)
    # --- codec alone
    tc, pay = med(lambda: core.frame_payloads(x, bits, F_file), 3)
    res['codec only'] = dict(enc=raw / tc / 1e6, size=sum(4 + len(b) for _, b in pay) / raw)
    # --- naive: codec + Ascon, no padding (one AEAD over the concatenated frames)
    def naive():
        p = core.frame_payloads(x, bits, F_file)
        body = b''.join(np.uint32(lf).tobytes() + b for lf, b in p)
        return ascon.encrypt(K, nn, b'', body)
    te, ct = med(naive, 3)
    res['codec + Ascon (no padding)'] = dict(enc=raw / te / 1e6, size=len(ct) / raw)
    # --- SHALE file mode
    for aead in ('ascon', 'aesgcm'):
        te, (blob, j) = med(lambda: core.encrypt_file(K, x, sr, bits, F_file, file_ratios, aead=aead), 3)
        td, y = med(lambda: core.decrypt_file(K, blob, file_ratios), 3)
        assert np.array_equal(y, x)
        res[f'SHALE file ({aead})'] = dict(enc=raw / te / 1e6, dec=raw / td / 1e6, size=len(blob) / raw, bucket=int(j))
    # --- SHALE stream mode
    te, (h, recs, idx) = med(lambda: core.encrypt_stream(K, x, sr, bits, F_stream, stream_buckets), 3)
    td, y = med(lambda: core.decrypt_stream(K, h, recs), 3)
    assert np.array_equal(y, x)
    nrec = len(recs)
    res[f'SHALE stream (ascon, {1000 * F_stream / sr:.0f} ms)'] = dict(
        enc=raw / te / 1e6, dec=raw / td / 1e6, size=(len(h) + sum(map(len, recs))) / raw,
        per_frame_enc_ms=1000 * te / nrec, per_frame_dec_ms=1000 * td / nrec)
    # cost breakdown for SHALE file (ascon)
    pay = core.frame_payloads(x, bits, F_file)
    body = sum(4 + len(b) for _, b in pay)
    ta, _ = med(lambda: ascon.encrypt(K, nn, b'', bytes(int(np.ceil(body * 1.1)))))
    res['breakdown'] = dict(codec_s=tc, aead_s=ta)
    print(name, json.dumps({k: v for k, v in res.items()}, indent=0, default=float)[:3000], flush=True)
    return res


if __name__ == '__main__':
    vctk = pickle.load(open(f'{R}/len_vctk.pkl', 'rb'))
    out = {}
    # speech: stream buckets from the VCTK profile, 20 ms frames
    top = core.frame_top(320, 1, 16)
    sb = eclq.design(np.concatenate([d[320] for d in vctk]), top, 4, 0.0, 4)
    fr = json.load(open(f'{R}/file_ratios.json'))
    x, sr, bits = load('data/CREMA-D/AudioWAV/*.wav', 60)
    out['speech'] = bench('speech', x, sr, bits, 4096, 320, sb, fr['K4'])
    for nm, pat in (('music16', 'data/bach16/*.wav'), ('music24', 'data/bach24/*.wav')):
        x, sr, bits = load(pat, 60)
        F = 4096
        top = core.frame_top(882, 2, bits)        # 20 ms at 44.1 kHz
        out[nm] = bench(nm, x, sr, bits, F, 882, eclq.uniform_buckets(top, 4), fr['K4'])
    json.dump(out, open(f'{R}/bench.json', 'w'), indent=1, default=float)
