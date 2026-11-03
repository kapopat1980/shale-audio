"""Ciphertext statistics, NIST SP 800-22 subset, sensitivity, tamper and packet-loss tests for SHALE."""
import sys, os, glob, json, pickle, time, numpy as np, soundfile as sf
sys.path.insert(0, '.'); sys.path.insert(0, 'exp')
from shale import core, eclq
import nist

R = 'results'
FR = json.load(open(f'{R}/file_ratios.json'))['K4']
rng = np.random.default_rng(2026)


def read(f):
    info = sf.info(f)
    bits = 24 if info.subtype == 'PCM_24' else 16
    x, sr = sf.read(f, dtype='int32')
    x = (x >> 16) if bits == 16 else (x >> 8)
    if x.ndim == 1:
        x = x[:, None]
    return x.astype(np.int64), sr, bits


def H(a, nb):
    c = np.bincount(a, minlength=nb).astype(float)
    p = c[c > 0] / c.sum()
    return float(-(p * np.log2(p)).sum())


def cstats(ct):
    b = np.frombuffer(ct, np.uint8)
    w = np.frombuffer(ct[: len(ct) // 2 * 2], '<i2').astype(float)
    c = np.bincount(b, minlength=256)
    e = len(b) / 256
    ideal = np.frombuffer(os.urandom(len(ct)), np.uint8)
    wi = np.frombuffer(ideal.tobytes()[: len(ct) // 2 * 2], '<u2')
    return dict(H8=H(b, 256), H16=H(np.frombuffer(ct[: len(ct) // 2 * 2], '<u2'), 65536),
                H16_ideal=H(wi, 65536), chi2=float(((c - e) ** 2 / e).sum()),
                r_adj=float(np.corrcoef(w[:-1], w[1:])[0, 1]), nbytes=len(ct))


def corpus():
    sets = {
        'CREMA-D speech': rng.choice(sorted(glob.glob('data/CREMA-D/AudioWAV/*.wav')), 40, replace=False),
        'MS-SNSD noise (test)': sorted(glob.glob('data/MS-SNSD/noise_test/*.wav'))[:20],
        'Bach10 stereo 16-bit': sorted(glob.glob('data/bach16/*.wav')),
        'Bach10 stereo 24-bit': sorted(glob.glob('data/bach24/*.wav')),
    }
    out = {}
    K = os.urandom(16)
    for name, fs in sets.items():
        rows = []
        for f in fs:
            x, sr, bits = read(f)
            blob, j = core.encrypt_file(K, x, sr, bits, 4096, FR)
            y = core.decrypt_file(K, blob, FR)
            st = cstats(blob[core.HDR_LEN:])
            # plaintext statistics for reference
            pcm = core.pcm_bytes(x, bits)
            xs = x[:, 0].astype(float)
            st.update(exact=bool(np.array_equal(x, y)), bucket=int(j), size_ratio=len(blob) / len(pcm),
                      H8_plain=H(np.frombuffer(pcm, np.uint8), 256),
                      r_adj_plain=float(np.corrcoef(xs[:-1], xs[1:])[0, 1]) if xs.std() > 0 else 1.0)
            rows.append(st)
        out[name] = rows
        a = lambda k: [r[k] for r in rows]
        print(name, len(rows), 'exact', all(a('exact')), 'H8 %.4f-%.4f' % (min(a('H8')), max(a('H8'))),
              'chi2 max %.1f' % max(a('chi2')), '|r| max %.4f' % max(np.abs(a('r_adj'))),
              'size %.3f' % np.mean(a('size_ratio')), flush=True)
    return out


def nist_run(nseq=1000):
    """Ciphertext bit streams: (a) speech/music/noise files encrypted with fresh salts,
    (b) digital silence (all-zero PCM) encrypted with fresh salts."""
    K = os.urandom(16)
    fs = sorted(glob.glob('data/CREMA-D/AudioWAV/*.wav')) + sorted(glob.glob('data/bach16/*.wav')) \
        + sorted(glob.glob('data/MS-SNSD/noise_test/*.wav'))
    rng.shuffle(fs)
    need = nseq * 125000
    res = {}
    for name in ('audio', 'silence'):
        buf = bytearray()
        i = 0
        while len(buf) < need:
            if name == 'audio':
                x, sr, bits = read(fs[i % len(fs)]); i += 1
            else:
                x, sr, bits = np.zeros((16000 * 10, 1), np.int64), 16000, 16
            blob, _ = core.encrypt_file(K, x, sr, bits, 4096, FR)
            buf += blob[core.HDR_LEN:]
        P = []
        for s in range(nseq):
            b = np.unpackbits(np.frombuffer(bytes(buf[s * 125000:(s + 1) * 125000]), np.uint8))
            P.append(nist.battery(b))
        P = np.array(P)
        summ, lo = nist.summarize(P)
        res[name] = dict(tests=nist.TESTS, prop=[s[0] for s in summ], pT=[s[1] for s in summ], min_prop=lo, nseq=nseq)
        print('NIST', name, [f'{p:.3f}/{t:.3f}' for p, t in summ], 'min', lo, flush=True)
    return res


def sensitivity(ntrial=20):
    fs = rng.choice(sorted(glob.glob('data/CREMA-D/AudioWAV/*.wav')), 10, replace=False)
    out = dict(fresh=[], reuse=[], key=[], wrongkey_rejected=0, n=0)
    for f in fs:
        x, sr, bits = read(f)
        for t in range(ntrial // 10 * 2):
            K = os.urandom(16)
            salt = os.urandom(16)
            x2 = x.copy()
            i = rng.integers(len(x))
            x2[i, 0] ^= 1                                       # flip LSB of one sample
            c1, _ = core.encrypt_file(K, x, sr, bits, 4096, FR, salt=salt)
            c2, _ = core.encrypt_file(K, x2, sr, bits, 4096, FR)  # fresh salt (normal use)
            c3, _ = core.encrypt_file(K, x2, sr, bits, 4096, FR, salt=salt)  # same salt (misuse)
            Kf = bytearray(K); Kf[rng.integers(16)] ^= 1 << int(rng.integers(8)); Kf = bytes(Kf)
            c4, _ = core.encrypt_file(Kf, x, sr, bits, 4096, FR, salt=salt)  # one key bit flipped
            for key, c in (('fresh', c2), ('reuse', c3), ('key', c4)):
                a = np.frombuffer(c1[core.HDR_LEN:], np.uint8); b = np.frombuffer(c[core.HDR_LEN:], np.uint8)
                n = min(len(a), len(b))
                aw = np.frombuffer(a[: n // 2 * 2].tobytes(), '<u2').astype(float); bw = np.frombuffer(b[: n // 2 * 2].tobytes(), '<u2').astype(float)
                out[key].append(dict(npcr8=float(np.mean(a[:n] != b[:n]) * 100), uaci8=float(np.mean(np.abs(a[:n].astype(float) - b[:n])) / 255 * 100),
                                     nscr16=float(np.mean(aw != bw) * 100), uaci16=float(np.mean(np.abs(aw - bw)) / 65535 * 100),
                                     same_len=len(a) == len(b)))
            try:
                core.decrypt_file(Kf, c1, FR)
            except core.AuthError:
                out['wrongkey_rejected'] += 1
            out['n'] += 1
    for k in ('fresh', 'reuse', 'key'):
        v = out[k]
        print(k, 'NPCR8 %.4f UACI8 %.3f NSCR16 %.4f UACI16 %.3f' % tuple(np.mean([r[m] for r in v]) for m in ('npcr8', 'uaci8', 'nscr16', 'uaci16')),
              'min NSCR16 %.4f' % min(r['nscr16'] for r in v), flush=True)
    print('wrong key rejected', out['wrongkey_rejected'], '/', out['n'])
    return out


def tamper(nflip=1000):
    K = os.urandom(16)
    x, sr, bits = read(sorted(glob.glob('data/CREMA-D/AudioWAV/*.wav'))[5])
    blob, _ = core.encrypt_file(K, x, sr, bits, 4096, FR)
    rej = 0
    for t in range(nflip):
        b = bytearray(blob)
        p = int(rng.integers(len(b))); b[p] ^= 1 << int(rng.integers(8))
        try:
            core.decrypt_file(K, bytes(b), FR)
        except Exception:
            rej += 1
    vctk = pickle.load(open(f'{R}/len_vctk.pkl', 'rb'))
    sb = eclq.design(np.concatenate([d[320] for d in vctk]), core.frame_top(320, 1, 16), 4, 0.0, 4)
    h, recs, idx = core.encrypt_stream(K, x, sr, bits, 320, sb)
    srej = 0
    for t in range(nflip):
        r = list(recs); i = int(rng.integers(len(r))); b = bytearray(r[i]); p = int(rng.integers(len(b))); b[p] ^= 1 << int(rng.integers(8)); r[i] = bytes(b)
        try:
            core.decrypt_stream(K, h, r)
        except core.AuthError:
            srej += 1
    ops = {}
    def chk(fn):
        try:
            fn(); return False
        except Exception:
            return True
    r = list(recs); r[1], r[2] = r[2], r[1]; ops['swap two records'] = chk(lambda: core.decrypt_stream(K, h, r))
    ops['drop last record'] = chk(lambda: core.decrypt_stream(K, h, recs[:-1]))
    ops['duplicate a record'] = chk(lambda: core.decrypt_stream(K, h, recs[:3] + recs[2:-1]))
    hb = bytearray(h); hb[8] ^= 1; ops['change sample rate in header'] = chk(lambda: core.decrypt_stream(K, bytes(hb), recs))
    h2, recs2, _ = core.encrypt_stream(K, x, sr, bits, 320, sb)
    r = list(recs); r[4] = recs2[4]; ops['splice record from another stream'] = chk(lambda: core.decrypt_stream(K, h, r))
    print('file flips rejected', rej, '/', nflip, 'stream flips rejected', srej, '/', nflip, ops)
    return dict(file_flips=nflip, file_rejected=rej, stream_flips=nflip, stream_rejected=srej, ops=ops)


def packet_loss():
    """stream mode, 20 ms records: independent decoding of surviving records"""
    vctk = pickle.load(open(f'{R}/len_vctk.pkl', 'rb'))
    sb = eclq.design(np.concatenate([d[320] for d in vctk]), core.frame_top(320, 1, 16), 4, 0.0, 4)
    K = os.urandom(16)
    fs = rng.choice(sorted(glob.glob('data/CREMA-D/AudioWAV/*.wav')), 20, replace=False)
    out = {}
    for p in (0.01, 0.05):
        kept = []; snr = []
        for f in fs:
            x, sr, bits = read(f)
            h, recs, idx = core.encrypt_stream(K, x, sr, bits, 320, sb)
            # receiver decrypts each record independently (helper reuses the stream format)
            p_ = core.parse_header(h); hd = core.header_digest(h)
            E = core.AEADS[p_['aead']](core.derive_key(K, p_['salt'], p_['aead']))
            y = np.zeros_like(x); ok = 0; n = len(recs)
            for i, ct in enumerate(recs):
                if rng.random() < p:
                    continue
                fin = 1 if i == n - 1 else 0
                pt = E.dec(i, fin, hd + np.uint64(i).tobytes() + bytes([fin]), ct)
                lf = int.from_bytes(pt[:4], 'little'); m = min(320, len(x) - i * 320)
                if lf & core.VERB:
                    y[i * 320:i * 320 + m] = core.pcm_from_bytes(pt[4:4 + (lf & ~core.VERB)], bits, m, 1)
                else:
                    o = np.zeros((m, 1), np.int64)
                    from shale import codec
                    codec.decode_frame(np.frombuffer(pt[4:4 + lf] + bytes(8), np.uint8), m, 1, bits, o)
                    y[i * 320:i * 320 + m] = o
                ok += 1
                assert np.array_equal(y[i * 320:i * 320 + m], x[i * 320:i * 320 + m])
            kept.append(ok / n)
            e = (x - y).astype(float); snr.append(10 * np.log10(np.sum(x.astype(float) ** 2) / max(np.sum(e ** 2), 1)))
        out[p] = dict(kept=float(np.mean(kept)), snr_db=float(np.mean(snr)))
        print('loss', p, out[p], flush=True)
    return out


if __name__ == '__main__':
    which = sys.argv[1:] or ['corpus', 'sens', 'tamper', 'loss', 'nist']
    path = f'{R}/stats.json'
    res = json.load(open(path)) if os.path.exists(path) else {}
    for w in which:
        t = time.time()
        res[w] = dict(corpus=corpus, sens=sensitivity, tamper=tamper, loss=packet_loss, nist=nist_run)[w]()
        json.dump(res, open(path, 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
        print(w, 'done', round(time.time() - t, 1), 's', flush=True)
