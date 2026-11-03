"""Stereo mixes of the Bach10 stems (constant-power panning), rendered at 16 and 24 bit with TPDF dither."""
import glob, os, numpy as np, soundfile as sf
pan = {'violin': -0.6, 'clarinet': -0.2, 'saxphone': 0.2, 'bassoon': 0.6}
rng = np.random.default_rng(10)
os.makedirs('data/bach16', exist_ok=True); os.makedirs('data/bach24', exist_ok=True)
for d in sorted(glob.glob('data/Bach10/[0-9]*')):
    name = os.path.basename(d)
    mix = None
    for inst, p in pan.items():
        x, sr = sf.read(f'{d}/{name}-{inst}.wav', dtype='float64')
        th = (p + 1) * np.pi / 4
        st = np.stack([x * np.cos(th), x * np.sin(th)], 1)
        mix = st if mix is None else mix[:len(st)] + st[:len(mix)]
    mix *= 0.89 / np.abs(mix).max()
    for bits, out in ((16, 'bach16'), (24, 'bach24')):
        q = 2 ** (bits - 1)
        tpdf = (rng.random(mix.shape) - rng.random(mix.shape))
        y = np.clip(np.round(mix * (q - 1) + tpdf), -q, q - 1).astype(np.int32)
        sf.write(f'data/{out}/{name}.wav', y if bits == 24 else y.astype(np.int16), sr, subtype=f'PCM_{bits}')
print('ok')
